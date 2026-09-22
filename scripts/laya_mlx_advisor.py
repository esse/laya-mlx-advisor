#!/usr/bin/env python3
"""Launch local Codex or Claude Code through one shared Laya routing daemon."""
import argparse
import concurrent.futures
import copy
import hmac
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

LEVELS = ("low", "medium", "high", "xhigh", "max")
ROOT = Path(__file__).resolve().parents[1]
# Anthropic's documented effort-capable models; unknown models pass through.
CLAUDE_MAX = re.compile(r"^claude-(?:(?:opus-(?:4-[678]|5)|sonnet-(?:4-6|5)|(?:fable|mythos)-5(?:-1)?|mythos-preview))(?:-\d{8})?$")
QUESTION = {"effort": {
    "type": "choice",
    "instructions": "Classify the reasoning needed for the NEXT coding step using the current evidence.",
    "criteria": {
        "low": "Mechanical edit, known command, file lookup, or reporting a confirmed result.",
        "medium": "Routine implementation with a clear approach and a few local decisions.",
        "high": "Nontrivial debugging or implementation requiring several connected reasoning steps.",
        "xhigh": "Subtle failures, conflicting evidence, or complex architectural and concurrency interactions.",
        "max": "Exceptionally difficult novel algorithms, research problems, or rigorous correctness proofs.",
    },
}}
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length"}


def note(message):
    print(f"[laya-mlx-advisor] {message}", file=sys.stderr, flush=True)


def excerpt(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit // 2] + " … " + text[-(limit // 2 - 3):]


def item_text(item):
    if item.get("type") in ("function_call_output", "custom_tool_call_output"):
        content = item.get("output", "")
    elif item.get("type") in ("function_call", "custom_tool_call"):
        content = f"{item.get('name', '')}: {item.get('arguments', item.get('input', ''))}"
    elif item.get("role") in ("user", "assistant"):
        content = item.get("content", "")
    else:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content
                         if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"))
    return ""


def context_for(body):
    items = body.get("input", [])
    if isinstance(items, str):
        return excerpt(items, 1200)
    recent, task = [], ""
    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        text = item_text(item)
        if not text:
            continue
        if item.get("role") == "user":
            task = excerpt(text, 400)
            break
        if len(recent) < 2:
            recent.append(f"{item.get('type', item.get('role'))}: {excerpt(text, 350)}")
    # ponytail: short excerpts fit Laya's encoder; use evaluated summaries if context loss hurts routing.
    return ("Latest evidence (newest first):\n" + "\n".join(recent) + "\nUser task:\n" + task) if recent or task else ""


def claude_context(body):
    items = []
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            items.append(message)
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                items.append({"role": message.get("role"), "content": block.get("text", "")})
            elif kind == "tool_result":
                items.append({"type": "function_call_output", "output": block.get("content", "")})
            elif kind == "tool_use":
                items.append({"type": "function_call", "name": block.get("name", ""),
                              "arguments": json.dumps(block.get("input", {}))})
    return context_for({"input": items})


def has_media(value):
    if isinstance(value, dict):
        return value.get("type") in ("image", "input_image", "input_audio", "input_file", "document", "audio") or any(
            has_media(v) for k, v in value.items() if k in ("content", "output"))
    return isinstance(value, list) and any(has_media(v) for v in value)


class Router:
    def __init__(self, agent, models, threshold=0.7, timeout=2.0):
        self.agent, self.models = agent, models
        self.threshold, self.timeout = threshold, timeout
        # ponytail: one inference worker serializes MLX; busy requests retain their original effort.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.pending = None

    def rewrite(self, body, protocol="codex", threshold=None, capabilities=None):
        model = body.get("model", "")
        if protocol == "claude":
            levels = list(LEVELS) if CLAUDE_MAX.fullmatch(model) else []
            if re.fullmatch(r"claude-(?:opus-4-6|sonnet-4-6|mythos-preview)(?:-\d{8})?", model):
                levels = [level for level in levels if level != "xhigh"]
            if re.fullmatch(r"claude-opus-4-5(?:-\d{8})?", model):
                levels = ["low", "medium", "high"]
            if body.get("thinking", {}).get("type") == "disabled":
                levels = [level for level in levels if level not in ("max", "xhigh")]
        else:
            levels = (capabilities or self.models).get(model, [])
        available = [level for level in LEVELS if level in levels]
        items = body.get("messages" if protocol == "claude" else "input", [])
        if has_media(items):
            note("unchanged: non-text context")
            return body
        if not available or "low" not in available:
            note("unchanged: model capabilities unavailable")
            return body
        if isinstance(items, list) and any(isinstance(i, dict) and
                (i.get("type") == "configuration_update" or "output_config" in i) for i in items):
            note("unchanged: per-message effort override in history")
            return body
        state = claude_context(body) if protocol == "claude" else context_for(body)
        if not state:
            return body
        start = time.monotonic()
        try:
            with self.lock:
                if self.pending is not None and not self.pending.done():
                    note("unchanged: classifier busy")
                    return body
                self.pending = self.executor.submit(self.agent.predict, state, QUESTION)
                future = self.pending
            answer = future.result(timeout=self.timeout)["answers"]["effort"]
            confidence = float(answer["probabilities"][answer["choice"]])
            if answer["choice"] not in LEVELS or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("invalid prediction")
            if confidence < (self.threshold if threshold is None else threshold):
                note(f"unchanged: confidence={confidence:.2f}")
                return body
            effort = next((level for level in available
                           if LEVELS.index(level) >= LEVELS.index(answer["choice"])), available[-1])
            result = copy.deepcopy(body)
            key = "output_config" if protocol == "claude" else "reasoning"
            result[key] = {**(result.get(key) or {}), "effort": effort}
            note(f"harness={protocol} effort={effort} probability={confidence:.2f} classifier_ms={(time.monotonic()-start)*1000:.0f}")
            return result
        except Exception as exc:
            # Exception messages may contain input data, so log only their class.
            note(f"unchanged: classifier {type(exc).__name__}")
            return body

    def close(self):
        # The daemon must retain its lifetime lock until the model worker has stopped.
        self.executor.shutdown(wait=True, cancel_futures=True)


def forwarded_headers(headers):
    blocked = HOP_HEADERS | {v.strip().lower() for v in headers.get("Connection", "").split(",")}
    return {k: v for k, v in headers.items() if k.lower() not in blocked and not k.lower().startswith("x-laya-advisor-")}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            if self.authorized():
                self.json_reply(200, {k: v for k, v in self.server.info.items() if k != "token"})
            return
        self.forward()

    def do_POST(self):
        if self.path == "/shutdown":
            if self.authorized():
                self.json_reply(200, {"stopping": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        self.forward()

    def json_reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        self.connection.settimeout(30)
        if not hmac.compare_digest(self.headers.get("X-Laya-Advisor-Token", ""), self.server.token):
            self.close_connection = True
            self.send_error(403, "Local proxy token required")
            return False
        return True

    def forward(self):
        self.close_connection = True
        if not self.authorized():
            return
        path = urlsplit(self.path)
        protocol = "codex"
        target = self.server.upstream
        forward_path = self.path
        if target is None:
            prefix, _, rest = path.path.lstrip("/").partition("/")
            base = self.server.upstreams.get(prefix)
            if base is None:
                self.send_error(404, "Unknown provider")
                return
            target = urlsplit(base)
            protocol = "claude" if prefix == "anthropic" else "codex"
            forward_path = "/" + rest + ("?" + path.query if path.query else "")
        endpoint = urlsplit(forward_path).path
        allowed = ("/v1/messages", "/v1/messages/count_tokens", "/v1/models") if protocol == "claude" else ("/responses", "/responses/compact", "/models")
        if path.scheme or path.netloc or endpoint not in allowed:
            self.send_error(404, "Unsupported endpoint")
            return
        if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding", "identity") != "identity":
            self.send_error(415, "Use uncompressed, Content-Length requests")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 <= size <= 64 * 1024 * 1024:
                raise ValueError()
        except ValueError:
            self.send_error(413, "Invalid or oversized request")
            return
        upstream = None
        started = False
        try:
            data = self.rfile.read(size)
            if len(data) != size:
                self.send_error(400, "Incomplete request")
                return
            if self.command == "POST" and endpoint in ("/responses", "/v1/messages"):
                try:
                    body = json.loads(data)
                    key = "output_config" if protocol == "claude" else "reasoning"
                    input_key = "messages" if protocol == "claude" else "input"
                    if (not isinstance(body, dict) or not isinstance(body.get(input_key, []), (list, str))
                            or not isinstance(body.get("model"), str)
                            or not isinstance(body.get(key, {}), (dict, type(None)))
                            or not isinstance(body.get("thinking", {}), dict)):
                        raise ValueError()
                    threshold = float(self.headers.get("X-Laya-Advisor-Threshold", "0.7"))
                    if not 0 <= threshold <= 1:
                        raise ValueError()
                    capabilities = json.loads(self.headers.get("X-Laya-Advisor-Capabilities", "null"))
                    if capabilities is not None and (not isinstance(capabilities, dict) or any(
                            not isinstance(v, list) or any(level not in LEVELS for level in v) for v in capabilities.values())):
                        raise ValueError()
                except (ValueError, UnicodeDecodeError):
                    self.send_error(400, "Invalid inference request")
                    return
                rewritten = self.server.router.rewrite(body, protocol, threshold, capabilities)
                if rewritten is not body:
                    data = json.dumps(rewritten, ensure_ascii=False).encode()
            connection = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
            upstream = connection(target.hostname, target.port, timeout=600)
            headers = forwarded_headers(self.headers)
            headers["Content-Length"] = str(len(data))
            headers["Connection"] = "close"
            upstream.request(self.command, target.path.rstrip("/") + forward_path, data, headers)
            response = upstream.getresponse()
            self.send_response(response.status)
            for key, value in forwarded_headers(response.headers).items():
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            started = True
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            note("upstream or client connection failed")
            if not started:
                self.send_error(502, "Upstream connection failed")
        finally:
            if upstream:
                upstream.close()


class Proxy(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, router, upstream, token):
        self.router, self.upstream, self.token = router, urlsplit(upstream) if upstream else None, token
        self.upstreams = {"openai": "https://api.openai.com/v1",
                          "chatgpt": "https://chatgpt.com/backend-api/codex",
                          "anthropic": "https://api.anthropic.com"}
        self.info = {}
        super().__init__(("127.0.0.1", 0), Handler)


def launch_args(base_url, token, auth, headers=None):
    settings = {
        "model_provider": "laya_mlx_advisor",
        "model_providers.laya_mlx_advisor.name": "Laya-MLX-Advisor",
        "model_providers.laya_mlx_advisor.base_url": base_url,
        "model_providers.laya_mlx_advisor.wire_api": "responses",
        "model_providers.laya_mlx_advisor.supports_websockets": False,
        "model_providers.laya_mlx_advisor.requires_openai_auth": auth == "chatgpt",
        "model_providers.laya_mlx_advisor.http_headers": {"X-Laya-Advisor-Token": token, **(headers or {})},
    }
    if auth == "api":
        settings["model_providers.laya_mlx_advisor.env_key"] = "OPENAI_API_KEY"
    return [arg for key, value in settings.items() for arg in ("-c", f"{key}=" +
            ("{" + ", ".join(f'{json.dumps(k)} = {json.dumps(v)}' for k, v in value.items()) + "}"
             if isinstance(value, dict) else json.dumps(value)))]


def claude_env(base_url, token, env=None, headers=None):
    env = dict(os.environ if env is None else env)
    if any(env.get(key) for key in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY")):
        raise ValueError("Claude Laya supports the direct Anthropic API, not cloud-provider transports")
    previous = env.get("ANTHROPIC_CUSTOM_HEADERS", "").splitlines()
    previous = [line for line in previous if not line.lower().startswith("x-laya-advisor-")]
    added = {"X-Laya-Advisor-Token": token, **(headers or {})}
    env.update(ANTHROPIC_BASE_URL=base_url, _CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL="1",
               ANTHROPIC_CUSTOM_HEADERS="\n".join(previous + [f"{k}: {v}" for k, v in added.items()]))
    return env


def read_models():
    home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    try:
        models = json.loads((home / "models_cache.json").read_text())["models"]
        return {m["slug"]: [level["effort"] for level in m["supported_reasoning_levels"]
                            if level["effort"] in LEVELS] for m in models}
    except (OSError, ValueError, KeyError, TypeError):
        note("no model catalog; use --efforts with --target-model, or run Codex once to populate it")
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harness", choices=("codex", "claude", "status", "stop"))
    parser.add_argument("--auth", choices=("chatgpt", "api"), default="chatgpt", help="Codex authentication")
    parser.add_argument("--classifier", default="aac6fef/laya-mlx", help="Hugging Face ID or local checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.7, help="Minimum winning-class probability")
    parser.add_argument("--target-model", help="Optional harness model; required with --efforts")
    parser.add_argument("--efforts", help="Explicit Codex capabilities, e.g. low,medium,high,xhigh,max")
    options, separator, forwarded = [], False, []
    for arg in sys.argv[1:]:
        if separator:
            forwarded.append(arg)
        elif arg == "--":
            separator = True
        else:
            options.append(arg)
    args = parser.parse_args(options)
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be 0..1")
    from laya_service import Client
    if args.harness in ("status", "stop"):
        client = Client.existing(args.classifier)
        if not client:
            print("Laya daemon is stopped.")
            return 0
        print(json.dumps(client.request("/shutdown" if args.harness == "stop" else "/health")))
        return 0
    if args.harness == "codex" and args.auth == "api" and not os.environ.get("OPENAI_API_KEY"):
        parser.error("--auth api requires OPENAI_API_KEY")
    headers = {"X-Laya-Advisor-Threshold": str(args.threshold)}
    if args.harness == "codex":
        models = read_models()
        if args.efforts:
            levels = args.efforts.split(",")
            if not args.target_model or "low" not in levels or any(level not in LEVELS for level in levels):
                parser.error("--efforts requires --target-model, low, and only low/medium/high/xhigh/max")
            models[args.target_model] = levels
        headers["X-Laya-Advisor-Capabilities"] = json.dumps(models, separators=(",", ":"))
    elif args.efforts:
        parser.error("--efforts applies only to Codex")
    note("connecting to the repository's shared Laya daemon")
    client = Client.connect(args.classifier)
    note(f"shared pid={client.info['pid']}; decisions: {ROOT / '.runtime/daemon.log'}")
    base = f"http://127.0.0.1:{client.info['port']}"
    if args.harness == "codex":
        route = "chatgpt" if args.auth == "chatgpt" else "openai"
        command = ["codex", *launch_args(base + "/" + route, client.info["token"], args.auth, headers)]
        env = dict(os.environ)
    else:
        command = ["claude", "--plugin-dir", str(ROOT / "plugins/laya-mlx-advisor")]
        env = claude_env(base + "/anthropic", client.info["token"], headers=headers)
    env["LAYA_MLX_ADVISOR_ROOT"] = str(ROOT)
    if args.target_model:
        command.extend(["--model", args.target_model])
    child = None
    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        child = subprocess.Popen([*command, *forwarded], env=env)
        # Both processes receive terminal Ctrl-C; let the harness cancel its turn.
        signal.signal(signal.SIGINT, lambda *_: None)
        signal.signal(signal.SIGTERM, lambda *_: child.terminate())
        return child.wait()
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if child and child.poll() is None:
            child.terminate()
            child.wait()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as exc:
        note(str(exc))
        sys.exit(1)
