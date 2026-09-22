"""Run with: python3 -m unittest -v"""
import copy
import fcntl
import http.client
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from scripts.laya_mlx_advisor import Router, Proxy, context_for, launch_args, claude_env, read_models


class Classifier:
    def __init__(self, choice="routine", confidence=0.99):
        self.choice, self.confidence = choice, confidence
        self.states = []

    def predict(self, state, questions):
        self.states.append(state)
        if isinstance(self.choice, Exception):
            raise self.choice
        return {"answers": {"effort": {
            "choice": self.choice, "confidence": 0.01,
            "probabilities": {self.choice: self.confidence},
        }}}


def request():
    return {"model": "test-model", "stream": True,
            "reasoning": {"effort": "high", "summary": "auto"},
            "input": [{"role": "user", "content": "Fix the parser"}],
            "tools": [{"type": "function", "name": "shell"}]}


class RoutingTests(unittest.TestCase):
    def test_real_catalog_filters_ultra_before_sending_capabilities(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "models_cache.json").write_text(json.dumps({"models": [
                {"slug": "gpt-6-astra", "supported_reasoning_levels": [
                    {"effort": e} for e in ("low", "high", "max", "ultra")]}]}))
            with patch.dict(os.environ, {"CODEX_HOME": folder}):
                self.assertEqual(read_models(), {"gpt-6-astra": ["low", "high", "max"]})

    def test_claude_rewrites_effort_each_request_without_changing_thinking_or_history(self):
        classifier = Classifier()
        router = Router(classifier, {})
        body = {"model": "claude-opus-4-8", "max_tokens": 4096, "stream": True,
                "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"},
                "messages": [{"role": "user", "content": "Fix the parser"}]}
        original = copy.deepcopy(body)
        self.assertEqual(router.rewrite(body, "claude")["output_config"]["effort"], "low")
        self.assertEqual(body, original)
        body["messages"].extend([
            {"role": "assistant", "content": [{"type": "tool_use", "name": "Bash", "id": "t1", "input": {"command": "test"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Unexplained race condition"}]},
        ])
        classifier.choice = "difficult"
        changed = router.rewrite(body, "claude")
        self.assertEqual(changed["output_config"]["effort"], "max")
        changed["output_config"]["effort"] = "high"
        self.assertEqual(changed, body)
        self.assertIn("Unexplained race", classifier.states[-1])
        self.assertIn("Fix the parser", classifier.states[-1])
        body["model"] = "claude-opus-5"
        body["thinking"] = {"type": "disabled"}
        self.assertEqual(router.rewrite(body, "claude")["output_config"]["effort"], "high")
        body["model"] = "unknown-claude"
        self.assertEqual(router.rewrite(body, "claude"), body)

    def test_claude_launch_preserves_credentials_and_other_custom_headers(self):
        env = claude_env("http://127.0.0.1:1234/anthropic", "secret", {
            "ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_CUSTOM_HEADERS": "X-Other: keep"})
        self.assertEqual(env["ANTHROPIC_API_KEY"], "test-key")
        self.assertIn("X-Other: keep", env["ANTHROPIC_CUSTOM_HEADERS"])
        self.assertIn("X-Laya-Advisor-Token: secret", env["ANTHROPIC_CUSTOM_HEADERS"])
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:1234/anthropic")

    def test_reclassifies_each_call_and_preserves_every_other_field(self):
        classifier = Classifier()
        router = Router(classifier, {"test-model": ["low", "high", "max"]})
        body = request()
        original = copy.deepcopy(body)
        self.assertEqual(router.rewrite(body)["reasoning"]["effort"], "low")
        self.assertEqual(body, original)
        classifier.choice = "difficult"
        body["input"].append({"type": "function_call_output", "call_id": "1",
                              "output": "Unexpected race; the same test fails intermittently"})
        result = router.rewrite(body)
        self.assertEqual(result["reasoning"]["effort"], "max")
        result["reasoning"]["effort"] = "high"
        self.assertEqual(result, body)
        self.assertIn("Unexpected race", classifier.states[-1])

    def test_uncertain_failed_unknown_and_configuration_updates_are_unchanged(self):
        body = request()
        for classifier in (Classifier(confidence=0.2), Classifier(choice=RuntimeError()),
                           Classifier(choice="invalid"), Classifier(confidence=float("nan"))):
            router = Router(classifier, {"test-model": ["low", "max"]})
            self.assertEqual(router.rewrite(body), body)
        router = Router(Classifier(), {})
        self.assertEqual(router.rewrite(body), body)
        body["input"].append({"type": "configuration_update", "reasoning": {"effort": "max"}})
        router = Router(Classifier(), {"test-model": ["low", "max"]})
        self.assertEqual(router.rewrite(body), body)

    def test_caps_at_supported_effort_without_selecting_ultra(self):
        for levels, expected in [(["low", "high", "xhigh"], "xhigh"),
                                 (["low", "high", "max", "ultra"], "max")]:
            router = Router(Classifier("difficult"), {"test-model": levels})
            self.assertEqual(router.rewrite(request())["reasoning"]["effort"], expected)

    def test_context_prioritizes_latest_tool_result_and_omits_hidden_reasoning(self):
        body = request()
        body["input"] = [
            {"role": "developer", "content": "SECRET INSTRUCTIONS"},
            {"role": "user", "content": "Fix race in parser"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Inspecting failures"}]},
            {"type": "reasoning", "encrypted_content": "SECRET CIPHERTEXT"},
            {"type": "function_call_output", "output": "start\n" + "x" * 10000 + "\nLATEST FAILURE"},
        ]
        state = context_for(body)
        self.assertLessEqual(len(state), 1600)
        self.assertIn("LATEST FAILURE", state)
        self.assertIn("Fix race", state)
        self.assertNotIn("SECRET", state)

    def test_launch_keeps_auth_in_codex_and_disables_websockets(self):
        args = launch_args("http://127.0.0.1:1234", "secret", "chatgpt")
        self.assertIn("model_providers.laya_mlx_advisor.supports_websockets=false", args)
        self.assertIn("model_providers.laya_mlx_advisor.requires_openai_auth=true", args)
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", args)

    def test_timeout_does_not_queue_more_predictions(self):
        release = threading.Event()
        classifier = Classifier()
        predict = classifier.predict
        def blocked(*args):
            release.wait(2)
            return predict(*args)
        classifier.predict = blocked
        router = Router(classifier, {"test-model": ["low", "max"]}, timeout=0.01)
        try:
            start = time.monotonic()
            self.assertEqual(router.rewrite(request()), request())
            self.assertEqual(router.rewrite(request()), request())
            self.assertLess(time.monotonic() - start, 1)
        finally:
            release.set()
            router.close()

    def test_shutdown_waits_for_inference_before_releasing_singleton(self):
        release, closed = threading.Event(), threading.Event()
        classifier = Classifier()
        predict = classifier.predict
        def blocked(*args):
            release.wait(5)
            return predict(*args)
        classifier.predict = blocked
        router = Router(classifier, {"test-model": ["low", "max"]}, timeout=0.01)
        self.assertEqual(router.rewrite(request()), request())
        def close():
            router.close()
            closed.set()
        thread = threading.Thread(target=close)
        thread.start()
        try:
            self.assertFalse(closed.wait(0.1), "Shutdown must retain the model lifetime lock during inference")
        finally:
            release.set()
            thread.join(2)
        self.assertTrue(closed.is_set())

    def test_images_and_empty_input_keep_original_effort(self):
        router = Router(Classifier(), {"test-model": ["low", "max"]})
        for content in ([], [{"role": "user", "content": [
                {"type": "input_text", "text": "Diagnose this"},
                {"type": "input_image", "image_url": "data:image/png;base64,abc"}]}]):
            body = request()
            body["input"] = content
            self.assertEqual(router.rewrite(body), body)

    def test_proxy_rewrites_only_responses_and_streams_errors_and_compaction(self):
        received = []

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received.append((self.path, dict(self.headers), json.loads(body)))
                status = 429 if self.path.endswith("compact") else 200
                payload = b'{"error":"slow down"}' if status == 429 else b'data: {"type":"response.completed"}\n\n'
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("x-request-id", "upstream-id")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        router = Router(Classifier(), {"test-model": ["low", "max"]})
        proxy = Proxy(router, f"http://127.0.0.1:{upstream.server_port}/v1", "secret")
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            for path, status in [("/responses", 200), ("/responses/compact", 429)]:
                conn = http.client.HTTPConnection("127.0.0.1", proxy.server_port)
                conn.request("POST", path, json.dumps(request()), {
                    "Content-Type": "application/json", "X-Laya-Advisor-Token": "secret",
                    "Authorization": "Bearer test", "ChatGPT-Account-Id": "account",
                })
                response = conn.getresponse()
                self.assertEqual(response.status, status)
                self.assertEqual(response.getheader("x-request-id"), "upstream-id")
                self.assertTrue(response.read())
                conn.close()
            self.assertEqual(received[0][2]["reasoning"]["effort"], "low")
            self.assertEqual(received[1][2], request())
            self.assertEqual(received[0][0], "/v1/responses")
            headers = {k.lower(): v for k, v in received[0][1].items()}
            self.assertNotIn("x-laya-advisor-token", headers)
            self.assertEqual(headers["authorization"], "Bearer test")
            conn = http.client.HTTPConnection("127.0.0.1", proxy.server_port)
            conn.request("POST", "/responses", "{}")
            self.assertEqual(conn.getresponse().status, 403)
            conn.close()
            self.assertEqual(len(received), 2)
            for data, status, extra in [("not json", 400, {}),
                    ('{"model":{},"input":[]}', 400, {}),
                    ('{"model":"test-model","reasoning":[]}', 400, {}),
                    ("{}", 415, {"Content-Encoding": "gzip"})]:
                conn = http.client.HTTPConnection("127.0.0.1", proxy.server_port)
                conn.request("POST", "/responses", data, {"X-Laya-Advisor-Token": "secret", **extra})
                response = conn.getresponse()
                self.assertEqual(response.status, status)
                response.read()
                conn.close()
            self.assertEqual(len(received), 2)
        finally:
            proxy.shutdown()
            proxy.server_close()
            router.close()
            upstream.shutdown()
            upstream.server_close()

    def test_stream_reaches_client_before_upstream_finishes(self):
        release = threading.Event()
        first, second = b"data: first\n\n", b"data: second\n\n"

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", str(len(first + second)))
                self.end_headers()
                self.wfile.write(first)
                self.wfile.flush()
                release.wait(5)
                self.wfile.write(second)
            def log_message(self, *args):
                pass

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        router = Router(Classifier(), {"test-model": ["low", "max"]})
        proxy = Proxy(router, f"http://127.0.0.1:{upstream.server_port}", "secret")
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=2)
        try:
            conn.request("POST", "/responses", json.dumps(request()), {"X-Laya-Advisor-Token": "secret"})
            response = conn.getresponse()
            self.assertEqual(response.read(len(first)), first)
            release.set()
            self.assertEqual(response.read(), second)
        finally:
            release.set()
            conn.close()
            proxy.shutdown()
            proxy.server_close()
            router.close()
            upstream.shutdown()
            upstream.server_close()

    def test_second_daemon_exits_before_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            (root / ".runtime").mkdir()
            shutil.copyfile("scripts/laya_service.py", root / "scripts/laya_service.py")
            with (root / ".runtime/daemon.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(["python3", str(root / "scripts/laya_service.py"), "--serve"],
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse((root / ".runtime/daemon.json").exists())

    @unittest.skipUnless(shutil.which("codex") and shutil.which("claude"), "Both harness CLIs required")
    def test_both_real_clis_change_effort_after_tool_result_on_one_proxy(self):
        """No cloud calls: both installed CLIs share one fake-upstream proxy."""
        received = []
        claude_received = []
        classifier = Classifier("difficult")

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path.startswith("/v1/messages"):
                    return self.claude_reply(body)
                received.append(body)
                n = len(received)
                if n == 1:
                    item = {"type": "custom_tool_call", "id": "fc_1", "call_id": "call_1",
                            "name": "exec", "namespace": "functions",
                            "input": "text(await tools.exec_command({cmd: \"printf 'confirmed result'\"}));"}
                    classifier.choice = "routine"
                else:
                    item = {"type": "message", "id": "msg_1", "role": "assistant",
                            "content": [{"type": "output_text", "text": "Verified routing.", "annotations": []}]}
                events = [
                    {"type": "response.created", "response": {"id": f"resp_{n}"}},
                    {"type": "response.output_item.done", "output_index": 0, "item": item},
                    {"type": "response.completed", "response": {"id": f"resp_{n}", "status": "completed",
                        "output": [item], "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}},
                ]
                payload = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def claude_reply(self, body):
                claude_received.append(body)
                n = len(claude_received)
                block = {"type": "tool_use", "id": "tool_1", "name": "Bash", "input": {}} if n == 1 else {"type": "text", "text": ""}
                delta = ({"type": "input_json_delta", "partial_json": json.dumps({"command": "printf 'confirmed result'", "description": "Verify result"})}
                         if n == 1 else {"type": "text_delta", "text": "Verified Claude routing."})
                events = [
                    {"type": "message_start", "message": {"id": f"msg_{n}", "type": "message", "role": "assistant",
                        "model": body["model"], "content": [], "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 0}}},
                    {"type": "content_block_start", "index": 0, "content_block": block},
                    {"type": "content_block_delta", "index": 0, "delta": delta},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta", "delta": {"stop_reason": "tool_use" if n == 1 else "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 10}},
                    {"type": "message_stop"},
                ]
                classifier.choice = "routine"
                payload = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        router = Router(classifier, {"gpt-6-astra": ["low", "high", "max"]})
        proxy = Proxy(router, None, "secret")
        proxy.upstreams["openai"] = f"http://127.0.0.1:{upstream.server_port}/v1"
        proxy.upstreams["anthropic"] = f"http://127.0.0.1:{upstream.server_port}"
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as home:
                env = {**os.environ, "CODEX_HOME": home, "OPENAI_API_KEY": "local-test-not-a-real-key"}
                headers = {"X-Laya-Advisor-Capabilities": json.dumps({"gpt-6-astra": ["low", "high", "max"]})}
                command = ["codex", *launch_args(f"http://127.0.0.1:{proxy.server_port}/openai", "secret", "api", headers),
                           "-m", "gpt-6-astra", "-s", "read-only", "-a", "never",
                           "exec", "--skip-git-repo-check", "--ephemeral", "-C", home,
                           "Run printf to verify the result, then report it."]
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Verified routing.", result.stdout)
                self.assertEqual([r["reasoning"]["effort"] for r in received], ["max", "low"])
                self.assertIn("confirmed result", classifier.states[-1])
                outputs = [i.get("output", "") for i in received[-1]["input"] if i.get("type") in ("function_call_output", "custom_tool_call_output")]
                self.assertTrue(any("confirmed result" in str(o) and ("exit_code\":0" in str(o) or "exit_code\": 0" in str(o) or "exit code: 0" in str(o).lower()) for o in outputs), outputs)
                classifier.choice = "difficult"
                env = claude_env(f"http://127.0.0.1:{proxy.server_port}/anthropic", "secret", {
                    **os.environ, "CLAUDE_CONFIG_DIR": home, "ANTHROPIC_API_KEY": "local-test-not-a-real-key",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})
                command = ["claude", "--bare", "--model", "claude-opus-4-8", "--no-session-persistence",
                           "--tools", "Bash", "--allowedTools", "Bash", "--permission-mode", "dontAsk",
                           "-p", "Run printf to verify the result, then report it."]
                result = subprocess.run(command, cwd=home, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertIn("Verified Claude routing.", result.stdout)
                self.assertEqual([r["output_config"]["effort"] for r in claude_received], ["max", "low"])
                tool_results = [b for m in claude_received[-1]["messages"] if isinstance(m["content"], list)
                                for b in m["content"] if b.get("type") == "tool_result"]
                self.assertTrue(any("confirmed result" in str(b.get("content")) and not b.get("is_error") for b in tool_results))
        finally:
            proxy.shutdown()
            proxy.server_close()
            router.close()
            upstream.shutdown()
            upstream.server_close()


if __name__ == "__main__":
    unittest.main()
