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

from scripts.laya_mlx_advisor import CLAUDE_BETA, Router, Proxy, context_for, launch_args, claude_env, read_models


class Classifier:
    def __init__(self, choice="low", confidence=0.99):
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
    return {"model": "gpt-6-astra", "stream": True,
            "reasoning": {"effort": "high", "summary": "auto"},
            "input": [{"role": "user", "content": "Fix the parser"}],
            "tools": [{"type": "function", "name": "shell"}]}


def route(router, body, protocol="codex", threshold=None, capabilities=None, conversation_id="session-1"):
    return router.rewrite(body, protocol, threshold, capabilities, conversation_id)


def effort_updates(body, protocol="codex"):
    key = "input" if protocol == "codex" else "messages"
    return [item for item in body[key] if item.get("type") == "configuration_update"
            or (protocol == "claude" and item.get("role") == "system" and "output_config" in item)]


class RoutingTests(unittest.TestCase):
    def test_supported_models_insert_cache_safe_update_before_new_items(self):
        classifier = Classifier("low")
        router = Router(classifier, {"gpt-6-astra": ["low", "high", "max"]})
        body = {"model": "gpt-6-astra", "reasoning": {"effort": "high"},
                "input": [{"role": "user", "content": "Fix it"},
                          {"type": "function_call", "call_id": "call-1", "name": "shell", "arguments": "{}"},
                          {"type": "function_call_output", "call_id": "call-1", "output": "done"}]}
        try:
            result = router.rewrite(body, conversation_id="session-1")
            self.assertEqual(result["reasoning"], {"effort": "high"})
            self.assertEqual(result["input"][2], {"type": "configuration_update",
                                                   "reasoning": {"effort": "low"}})
            self.assertEqual(result["input"][3], body["input"][2])
        finally:
            router.close()

    def test_unsupported_or_missing_conversation_requests_are_unchanged(self):
        classifier = Classifier("max")
        router = Router(classifier, {"gpt-6-astra": ["low", "max"]})
        try:
            body = request()
            self.assertIs(route(router, body, conversation_id=None), body)
            body["model"] = "gpt-5"
            self.assertIs(route(router, body, conversation_id="session-1"), body)
            claude = {"model": "claude-opus-5-5", "thinking": {"type": "adaptive"},
                      "output_config": {"effort": "high"},
                      "messages": [{"role": "user", "content": "Hello"}]}
            self.assertIs(router.rewrite(claude, "claude"), claude)
            self.assertFalse(classifier.states)
        finally:
            router.close()

    def test_replays_updates_across_tool_results_and_next_turns(self):
        classifier = Classifier("max")
        router = Router(classifier, {"gpt-6-astra": ["low", "max"]})
        body = request()
        body["reasoning"]["effort"] = "low"
        try:
            first = route(router, body)
            self.assertEqual([u["reasoning"]["effort"] for u in effort_updates(first)], ["max"])
            body["input"].extend([{"type": "function_call", "call_id": "call-1", "name": "shell", "arguments": "{}"},
                                  {"type": "function_call_output", "call_id": "call-1", "output": "old"}])
            classifier.choice = "low"
            second = route(router, body)
            self.assertEqual([u["reasoning"]["effort"] for u in effort_updates(second)], ["max", "low"])
            body["input"][-1]["output"] = "microcompacted"
            body["input"][-1]["cache_control"] = {"type": "ephemeral"}
            before = len(classifier.states)
            replay = route(router, body)
            self.assertEqual(len(classifier.states), before)
            self.assertEqual([u["reasoning"]["effort"] for u in effort_updates(replay)], ["max", "low"])
            body["input"].append({"role": "user", "content": "Continue"})
            classifier.choice = "max"
            third = route(router, body)
            self.assertEqual([u["reasoning"]["effort"] for u in effort_updates(third)], ["max", "low"])
            self.assertEqual(third["input"][3]["type"], "configuration_update")

            claude = {"model": "claude-opus-5-5", "thinking": {"type": "adaptive"},
                      "output_config": {"effort": "low"},
                      "messages": [{"role": "user", "content": "Start"}]}
            classifier.choice = "max"
            first = route(router, claude, "claude", conversation_id="claude-session")
            self.assertEqual([u["output_config"]["effort"] for u in effort_updates(first, "claude")], ["max"])
            claude["messages"].extend([
                {"role": "assistant", "content": [{"type": "tool_use", "id": "tool-1", "name": "shell", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "old"}]},
            ])
            classifier.choice = "low"
            second = route(router, claude, "claude", conversation_id="claude-session")
            self.assertEqual([u["output_config"]["effort"] for u in effort_updates(second, "claude")], ["max", "low"])
            claude["messages"][0]["cache_control"] = {"type": "ephemeral"}
            claude["messages"][-1]["content"][0]["content"] = "microcompacted"
            before = len(classifier.states)
            replay = route(router, claude, "claude", conversation_id="claude-session")
            self.assertEqual(len(classifier.states), before)
            self.assertEqual([u["output_config"]["effort"] for u in effort_updates(replay, "claude")], ["max", "low"])
            claude["messages"].append({"role": "user", "content": "Continue"})
            classifier.choice = "max"
            third = route(router, claude, "claude", conversation_id="claude-session")
            self.assertEqual([u["output_config"]["effort"] for u in effort_updates(third, "claude")], ["max", "low", "max"])
            self.assertEqual(third["messages"][-2]["role"], "system")
            claude["messages"][2]["content"][0]["tool_use_id"] = "rewritten-tool"
            classifier.choice = "low"
            rewritten = route(router, claude, "claude", conversation_id="claude-session")
            self.assertEqual([u["output_config"]["effort"] for u in effort_updates(rewritten, "claude")], ["max", "low"])
        finally:
            router.close()

    def test_fallback_and_equal_effort_are_decided_once_but_replay_old_updates(self):
        classifier = Classifier("max")
        router = Router(classifier, {"gpt-6-astra": ["low", "max"]})
        body = request()
        body["reasoning"]["effort"] = "low"
        try:
            route(router, body)
            body["input"].extend([{"type": "function_call", "call_id": "call-1", "name": "shell", "arguments": "{}"},
                                  {"type": "function_call_output", "call_id": "call-1", "output": "result"}])
            classifier.confidence = 0.2
            fallback = route(router, body)
            self.assertEqual([u["reasoning"]["effort"] for u in effort_updates(fallback)], ["max"])
            calls = len(classifier.states)
            self.assertEqual(route(router, body), fallback)
            self.assertEqual(len(classifier.states), calls)
            equal_classifier = Classifier("low")
            equal_router = Router(equal_classifier, {"gpt-6-astra": ["low", "max"]})
            try:
                equal_body = request()
                equal_body["reasoning"]["effort"] = "low"
                equal = route(equal_router, equal_body, conversation_id="equal")
                self.assertEqual(equal, equal_body)
                self.assertEqual(route(equal_router, equal_body, conversation_id="equal"), equal_body)
                self.assertEqual(len(equal_classifier.states), 1)
            finally:
                equal_router.close()
        finally:
            router.close()

    def test_history_branches_are_isolated_and_state_is_bounded(self):
        classifier = Classifier("low")
        router = Router(classifier, {"gpt-6-astra": ["low", "max"]})
        try:
            first = request()
            first["reasoning"]["effort"] = "high"
            branch_a = route(router, first, conversation_id="same-session")
            branch_b_body = request()
            branch_b_body["input"][0]["content"] = "Different branch"
            branch_b = route(router, branch_b_body, conversation_id="same-session")
            self.assertEqual(len(effort_updates(branch_a)), 1)
            self.assertEqual(len(effort_updates(branch_b)), 1)
            self.assertEqual(len(classifier.states), 2)
            for index in range(129):
                branch = request()
                branch["input"][0]["content"] = f"branch-{index}"
                route(router, branch, conversation_id=f"session-{index}")
            self.assertEqual(len(router.states), 128)
        finally:
            router.close()

    def test_per_key_record_cap_keeps_replay_bounded(self):
        classifier = Classifier("max")
        router = Router(classifier, {"gpt-6-astra": ["low", "max"]})
        body = request()
        body["reasoning"]["effort"] = "low"
        try:
            route(router, body, conversation_id="cap")
            for index in range(1, 70):
                body["input"].extend([{"type": "function_call", "call_id": str(index), "name": "shell", "arguments": "{}"},
                                       {"type": "function_call_output", "call_id": str(index), "output": "result"}])
                classifier.choice = "low" if index % 2 else "max"
                result = route(router, body, conversation_id="cap")
            self.assertEqual(len(effort_updates(result)), 64)
        finally:
            router.close()

    def test_concurrent_same_key_requests_do_not_duplicate_updates(self):
        started, release = threading.Event(), threading.Event()
        classifier = Classifier("max")
        predict = classifier.predict

        def blocked(*args):
            started.set()
            release.wait(2)
            return predict(*args)

        classifier.predict = blocked
        router = Router(classifier, {"gpt-6-astra": ["low", "max"]})
        body = request()
        body["reasoning"]["effort"] = "low"
        results = []

        def run():
            results.append(route(router, copy.deepcopy(body), conversation_id="concurrent"))

        first = threading.Thread(target=run)
        first.start()
        self.assertTrue(started.wait(1))
        second = threading.Thread(target=run)
        second.start()
        second.join(1)
        release.set()
        first.join(2)
        try:
            self.assertEqual(sum(len(effort_updates(result)) for result in results), 1)
        finally:
            router.close()

    def test_codex_bypasses_unsupported_modes_but_claude_context_clear_routes(self):
        router = Router(Classifier("max"), {"gpt-6-astra": ["low", "max"]})
        try:
            for field, value in (("context_management", {}), ("truncation", "auto"),
                                 ("previous_response_id", "resp-1")):
                body = request()
                body[field] = value
                self.assertIs(route(router, body, conversation_id=field), body)
            body = request()
            body["reasoning"]["other"] = "unsupported"
            self.assertIs(route(router, body, conversation_id="reasoning"), body)
            body = request()
            body["input"].append({"type": "compaction", "id": "c1"})
            self.assertIs(route(router, body, conversation_id="compaction"), body)
            body = request()
            body["truncation"] = "disabled"
            body["reasoning"]["effort"] = "low"
            routed = route(router, body, conversation_id="disabled-truncation")
            self.assertEqual(routed["input"][0]["reasoning"]["effort"], "max")
            claude = {"model": "claude-opus-5-5", "thinking": {"type": "adaptive"},
                      "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
                      "output_config": {"effort": "low"},
                      "messages": [{"role": "user", "content": "Route this"}]}
            routed = route(router, claude, "claude", conversation_id="claude-clear")
            self.assertEqual(routed["messages"][0]["output_config"]["effort"], "max")
        finally:
            router.close()

    def test_selects_every_effort_for_both_harnesses(self):
        levels = ["low", "medium", "high", "xhigh", "max"]
        for protocol in ("codex", "claude"):
            classifier = Classifier()
            router = Router(classifier, {"gpt-6-astra": levels})
            try:
                for level in levels:
                    with self.subTest(protocol=protocol, level=level):
                        classifier.choice = level
                        body = request() if protocol == "codex" else {
                            "model": "claude-opus-5-5", "messages": [{"role": "user", "content": "Implement this task"}],
                            "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
                        key = "reasoning" if protocol == "codex" else "output_config"
                        body[key]["effort"] = "low" if level != "low" else "high"
                        result = route(router, body, protocol, conversation_id=f"session-{level}")
                        self.assertEqual(result[key]["effort"], body[key]["effort"])
                        update = result["input" if protocol == "codex" else "messages"][0]
                        self.assertEqual(update.get("reasoning", update.get("output_config"))["effort"], level)
            finally:
                router.close()

    def test_missing_intermediate_level_rounds_up_without_exceeding_capabilities(self):
        classifier = Classifier("xhigh")
        router = Router(classifier, {"gpt-6-astra": ["low", "medium", "high", "max"]})
        try:
            result = route(router, request())
            self.assertEqual(result["reasoning"]["effort"], "high")
            self.assertEqual(result["input"][0]["reasoning"]["effort"], "max")
            for model in ("claude-opus-5-5", "claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5"):
                body = {"model": model, "messages": [{"role": "user", "content": "Investigate this task"}]}
                result = route(router, body, "claude", conversation_id=model)
                self.assertEqual(result["messages"][0]["output_config"]["effort"], "xhigh")
            classifier.choice = "medium"
            router.models = {"gpt-6-astra": ["low", "high"]}
            capped_body = request()
            capped_body["reasoning"]["effort"] = "max"
            result = route(router, capped_body, conversation_id="session-cap")
            self.assertEqual(result["input"][0]["reasoning"]["effort"], "high")
        finally:
            router.close()

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
        body = {"model": "claude-opus-5-5", "max_tokens": 4096, "stream": True,
                "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"},
                "messages": [{"role": "user", "content": "Fix the parser"}]}
        original = copy.deepcopy(body)
        first = route(router, body, "claude")
        self.assertEqual(first["output_config"]["effort"], "high")
        self.assertEqual(first["messages"][0]["output_config"]["effort"], "low")
        self.assertEqual(body, original)
        body["messages"].extend([
            {"role": "assistant", "content": [{"type": "tool_use", "name": "Bash", "id": "t1", "input": {"command": "test"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Unexplained race condition"}]},
        ])
        classifier.choice = "max"
        changed = route(router, body, "claude")
        self.assertEqual(changed["output_config"]["effort"], "high")
        self.assertEqual(changed["messages"][1], body["messages"][0])
        self.assertEqual(changed["messages"][2], body["messages"][1])
        self.assertEqual(changed["messages"][3]["output_config"]["effort"], "max")
        self.assertEqual(changed["messages"][4], body["messages"][2])
        self.assertIn("Unexplained race", classifier.states[-1])
        self.assertIn("Fix the parser", classifier.states[-1])
        body["model"] = "claude-opus-5"
        body["thinking"] = {"type": "disabled"}
        body["output_config"]["effort"] = "low"
        disabled = route(router, body, "claude", conversation_id="disabled")
        self.assertEqual(disabled["output_config"]["effort"], "low")
        self.assertEqual(disabled["messages"][2]["output_config"]["effort"], "high")
        body["model"] = "unknown-claude"
        self.assertEqual(route(router, body, "claude", conversation_id="unknown"), body)

    def test_claude_launch_preserves_credentials_and_other_custom_headers(self):
        env = claude_env("http://127.0.0.1:1234/anthropic", "secret", {
            "ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_CUSTOM_HEADERS": "X-Other: keep"})
        self.assertEqual(env["ANTHROPIC_API_KEY"], "test-key")
        self.assertIn("X-Other: keep", env["ANTHROPIC_CUSTOM_HEADERS"])
        self.assertIn("X-Laya-Advisor-Token: secret", env["ANTHROPIC_CUSTOM_HEADERS"])
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:1234/anthropic")

    def test_reclassifies_each_call_and_preserves_every_other_field(self):
        classifier = Classifier()
        router = Router(classifier, {"gpt-6-astra": ["low", "high", "max"]})
        body = request()
        original = copy.deepcopy(body)
        first = route(router, body)
        self.assertEqual(first["reasoning"]["effort"], "high")
        self.assertEqual(first["input"][0]["reasoning"]["effort"], "low")
        self.assertEqual(body, original)
        classifier.choice = "max"
        body["input"].extend([{"type": "function_call", "call_id": "1", "name": "shell", "arguments": "{}"},
                              {"type": "function_call_output", "call_id": "1",
                               "output": "Unexpected race; the same test fails intermittently"}])
        result = route(router, body)
        self.assertEqual(result["reasoning"]["effort"], "high")
        self.assertEqual(result["input"][0]["reasoning"]["effort"], "low")
        self.assertEqual(result["input"][3]["reasoning"]["effort"], "max")
        self.assertEqual(result["input"][4], body["input"][2])
        self.assertIn("Unexpected race", classifier.states[-1])

    def test_uncertain_failed_unknown_and_configuration_updates_are_unchanged(self):
        body = request()
        for classifier in (Classifier(confidence=0.2), Classifier(choice=RuntimeError()),
                           Classifier(choice="invalid"), Classifier(confidence=float("nan"))):
            router = Router(classifier, {"gpt-6-astra": ["low", "max"]})
            self.assertEqual(route(router, body), body)
            self.assertEqual(route(router, body), body)
            self.assertEqual(len(classifier.states), 1)
        router = Router(Classifier(), {})
        self.assertEqual(route(router, body), body)
        body["input"].append({"type": "configuration_update", "reasoning": {"effort": "max"}})
        router = Router(Classifier(), {"gpt-6-astra": ["low", "max"]})
        self.assertEqual(route(router, body), body)
        claude = {"model": "claude-opus-5-5", "thinking": {"type": "adaptive"},
                  "messages": [{"role": "system", "content": [], "output_config": {"effort": "low"}},
                               {"role": "user", "content": "Do not route"}]}
        self.assertIs(route(router, claude, "claude", conversation_id="claude-override"), claude)

    def test_caps_at_supported_effort_without_selecting_ultra(self):
        for levels, expected in [(["low", "high", "xhigh"], "xhigh"),
                                 (["low", "high", "max", "ultra"], "max")]:
            router = Router(Classifier("max"), {"gpt-6-astra": levels})
            result = route(router, request(), conversation_id=expected)
            self.assertEqual(result["input"][0]["reasoning"]["effort"], expected)

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
        router = Router(classifier, {"gpt-6-astra": ["low", "max"]}, timeout=0.01)
        try:
            start = time.monotonic()
            self.assertEqual(route(router, request()), request())
            self.assertEqual(route(router, request()), request())
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
        router = Router(classifier, {"gpt-6-astra": ["low", "max"]}, timeout=0.01)
        self.assertEqual(route(router, request()), request())
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
        router = Router(Classifier(), {"gpt-6-astra": ["low", "max"]})
        for content in ([], [{"role": "user", "content": [
                {"type": "input_text", "text": "Diagnose this"},
                {"type": "input_image", "image_url": "data:image/png;base64,abc"}]}]):
            body = request()
            body["input"] = content
            self.assertEqual(route(router, body), body)

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
        router = Router(Classifier(), {"gpt-6-astra": ["low", "max"]})
        proxy = Proxy(router, f"http://127.0.0.1:{upstream.server_port}/v1", "secret")
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            for path, status in [("/responses", 200), ("/responses/compact", 429)]:
                conn = http.client.HTTPConnection("127.0.0.1", proxy.server_port)
                conn.request("POST", path, json.dumps(request()), {
                    "Content-Type": "application/json", "X-Laya-Advisor-Token": "secret",
                    "Authorization": "Bearer test", "ChatGPT-Account-Id": "account", "session-id": "session-1",
                })
                response = conn.getresponse()
                self.assertEqual(response.status, status)
                self.assertEqual(response.getheader("x-request-id"), "upstream-id")
                self.assertTrue(response.read())
                conn.close()
            self.assertEqual(received[0][2]["reasoning"]["effort"], "high")
            self.assertEqual(received[0][2]["input"][0]["reasoning"]["effort"], "low")
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
                    ('{"model":"gpt-6-astra","reasoning":[]}', 400, {}),
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

    def test_claude_beta_header_is_added_only_for_proxy_updates(self):
        received = []

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(dict(self.headers))
                payload = b"{}"
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        router = Router(Classifier("low"), {})
        proxy = Proxy(router, None, "secret")
        proxy.upstreams["anthropic"] = f"http://127.0.0.1:{upstream.server_port}"
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            values = (None, "existing-beta", f"existing-beta,{CLAUDE_BETA},{CLAUDE_BETA}")
            for index, existing in enumerate(values):
                body = {"model": "claude-opus-5-5", "thinking": {"type": "adaptive"},
                        "output_config": {"effort": "high"},
                        "metadata": {"user_id": json.dumps({"session_id": f"beta-{index}"})},
                        "messages": [{"role": "user", "content": "Route this"}]}
                headers = {"Content-Type": "application/json", "X-Laya-Advisor-Token": "secret",
                           "X-Claude-Code-Session-Id": f"beta-{index}"}
                if existing is not None:
                    headers["Anthropic-Beta"] = existing
                conn = http.client.HTTPConnection("127.0.0.1", proxy.server_port)
                conn.request("POST", "/anthropic/v1/messages", json.dumps(body), headers)
                self.assertEqual(conn.getresponse().status, 200)
                conn.close()
            beta = [{key.lower(): value for key, value in headers.items()}.get("anthropic-beta")
                    for headers in received]
            self.assertEqual(beta[0].lower().count("mid-conversation-output-config-2026-07-01"), 1)
            self.assertEqual(beta[1], "existing-beta,mid-conversation-output-config-2026-07-01")
            self.assertEqual(beta[2], "existing-beta,mid-conversation-output-config-2026-07-01")
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
        router = Router(Classifier(), {"gpt-6-astra": ["low", "max"]})
        proxy = Proxy(router, f"http://127.0.0.1:{upstream.server_port}", "secret")
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=2)
        try:
            conn.request("POST", "/responses", json.dumps(request()),
                         {"X-Laya-Advisor-Token": "secret", "session-id": "stream-session"})
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
        classifier = Classifier("max")

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
                    classifier.choice = "low"
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
                classifier.choice = "low"
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
                original_effort = received[0]["reasoning"]["effort"]
                self.assertEqual([r["reasoning"]["effort"] for r in received], [original_effort, original_effort])
                first_updates = [item for item in received[0]["input"]
                                 if item.get("type") == "configuration_update"]
                second_updates = [item for item in received[1]["input"]
                                  if item.get("type") == "configuration_update"]
                self.assertEqual([item["reasoning"]["effort"] for item in first_updates], ["max"])
                self.assertEqual([item["reasoning"]["effort"] for item in second_updates], ["max", "low"])
                tool_indices = [i for i, item in enumerate(received[-1]["input"])
                                if item.get("type") in ("function_call_output", "custom_tool_call_output")]
                self.assertTrue(tool_indices)
                self.assertEqual(received[-1]["input"][tool_indices[-1] - 1]["type"], "configuration_update")
                self.assertIn("confirmed result", classifier.states[-1])
                outputs = [i.get("output", "") for i in received[-1]["input"] if i.get("type") in ("function_call_output", "custom_tool_call_output")]
                self.assertTrue(any("confirmed result" in str(o) and ("exit_code\":0" in str(o) or "exit_code\": 0" in str(o) or "exit code: 0" in str(o).lower()) for o in outputs), outputs)
                classifier.choice = "max"
                env = claude_env(f"http://127.0.0.1:{proxy.server_port}/anthropic", "secret", {
                    **os.environ, "CLAUDE_CONFIG_DIR": home, "ANTHROPIC_API_KEY": "local-test-not-a-real-key",
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"})
                command = ["claude", "--bare", "--model", "claude-opus-5-5", "--no-session-persistence",
                           "--tools", "Bash", "--allowedTools", "Bash", "--permission-mode", "dontAsk",
                           "-p", "Run printf to verify the result, then report it."]
                result = subprocess.run(command, cwd=home, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertIn("Verified Claude routing.", result.stdout)
                original_effort = claude_received[0]["output_config"]["effort"]
                self.assertEqual([r["output_config"]["effort"] for r in claude_received],
                                 [original_effort, original_effort])
                first_updates = [m for m in claude_received[0]["messages"]
                                 if m.get("content") == [] and m.get("output_config")]
                second_updates = [m for m in claude_received[1]["messages"]
                                  if m.get("content") == [] and m.get("output_config")]
                self.assertEqual(first_updates, [])
                self.assertEqual([message["output_config"]["effort"] for message in second_updates], ["low"])
                final_tool_result = [i for i, message in enumerate(claude_received[-1]["messages"])
                                     if any(isinstance(block, dict) and block.get("type") == "tool_result"
                                            for block in message.get("content", []))][-1]
                self.assertEqual(claude_received[-1]["messages"][final_tool_result - 1]["output_config"]["effort"], "low")
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
