#!/usr/bin/env python3
"""Launch local Codex or Claude Code through one shared Laya routing daemon."""
import argparse
import concurrent.futures
import copy
from collections import OrderedDict
import hashlib
import hmac
import http.client
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

LEVELS = ("low", "medium", "high", "xhigh", "max")
ROOT = Path(__file__).resolve().parents[1]
CLAUDE_MODELS = {"claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5-5", "claude-opus-5"}
CODEX_MODELS = {"gpt-6-astra", "gpt-6-sol", "gpt-6-luna"}
CLAUDE_BETA = "mid-conversation-output-config-2026-07-01"
STATE_LIMIT = 128
RECORD_LIMIT = 64
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


def normalized_item(value):
    if isinstance(value, dict):
        kind = value.get("type")
        if kind == "tool_result":
            return {"type": kind, "tool_use_id": value.get("tool_use_id")}
        if kind in ("function_call_output", "custom_tool_call_output"):
            return {"type": kind, "call_id": value.get("call_id")}
        return {key: normalized_item(item) for key, item in value.items() if key != "cache_control"}
    if isinstance(value, list):
        return [normalized_item(item) for item in value]
    return value


def item_hash(item):
    return hashlib.sha256(json.dumps(normalized_item(item), sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def append_claude_beta(headers):
    values, seen = [], set()
    for key in list(headers):
        if key.lower() != "anthropic-beta":
            continue
        for value in headers.pop(key).split(","):
            value = value.strip()
            if value and value.lower() not in seen:
                values.append(value)
                seen.add(value.lower())
    if CLAUDE_BETA.lower() not in seen:
        values.append(CLAUDE_BETA)
    headers["anthropic-beta"] = ",".join(values)


def claude_boundary(items):
    if not items or items[-1].get("role") != "user":
        return None
    content = items[-1].get("content")
    if isinstance(content, str):
        return len(items) - 1 if content else None
    if isinstance(content, list) and any(isinstance(block, dict) and
            block.get("type") in ("text", "tool_result") for block in content):
        return len(items) - 1
    return None


def codex_boundary(items):
    def is_new(item):
        return item.get("role") in ("user", "developer") or item.get("type") in (
            "function_call_output", "custom_tool_call_output")
    index = len(items)
    while index and is_new(items[index - 1]):
        index -= 1
    return index if index < len(items) else None


class Router:
    def __init__(self, agent, models, threshold=0.7, timeout=2.0):
        self.agent, self.models = agent, models
        self.threshold, self.timeout = threshold, timeout
        # ponytail: one inference worker serializes MLX; busy requests retain their original effort.
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.states = OrderedDict()
        self.pending = None

    def _conversation_id(self, body, protocol, conversation_id):
        if protocol == "claude":
            metadata = body.get("metadata")
            user_id = metadata.get("user_id") if isinstance(metadata, dict) else None
            if isinstance(user_id, str):
                try:
                    user_id = json.loads(user_id)
                except (TypeError, ValueError):
                    user_id = None
            if isinstance(user_id, dict) and user_id.get("session_id"):
                return user_id["session_id"]
        return conversation_id if isinstance(conversation_id, str) and conversation_id else None

    def _state_key(self, protocol, model, conversation_id, items):
        value = json.dumps([protocol, model, conversation_id, item_hash(items[0])],
                           separators=(",", ":"))
        # ponytail: first-item hashing separates normal branches; identical first items can still collide.
        return hashlib.sha256(value.encode()).hexdigest()

    def _normalize_state(self, state, items):
        records = state["records"]
        valid = []
        invalid_at = None
        for record in records:
            index, anchor, _ = record
            if index >= len(items) or item_hash(items[index]) != anchor:
                invalid_at = index
                break
            valid.append(record)
        decided = state["decided"]
        if invalid_at is not None:
            decided = {index: anchor for index, anchor in decided.items()
                       if index < invalid_at and index < len(items) and item_hash(items[index]) == anchor}
        else:
            decided = {index: anchor for index, anchor in decided.items()
                       if index < len(items) and item_hash(items[index]) == anchor}
        if valid != records or decided != state["decided"]:
            state["records"] = valid
            state["decided"] = decided
            state["version"] += 1
        return state

    def _state_snapshot(self, key, items):
        with self.state_lock:
            state = self.states.get(key)
            if state is None:
                state = {"records": [], "decided": {}, "version": 0}
                self.states[key] = state
                if len(self.states) > STATE_LIMIT:
                    self.states.popitem(last=False)
            self.states.move_to_end(key)
            self._normalize_state(state, items)
            return state["version"], list(state["records"]), dict(state["decided"])

    def _commit(self, key, items, index, version, records, decided, effort):
        with self.state_lock:
            state = self.states.get(key)
            if state is None:
                return records, False
            self._normalize_state(state, items)
            inserted = False
            if (state["version"] == version and state["records"] == records and
                    state["decided"] == decided and index not in state["decided"] and
                    not any(record[0] == index for record in state["records"])):
                if effort is not None and len(state["records"]) < RECORD_LIMIT:
                    state["records"].append((index, item_hash(items[index]), effort))
                    state["records"].sort(key=lambda record: record[0])
                    inserted = True
                state["decided"][index] = item_hash(items[index])
                state["version"] += 1
            self.states.move_to_end(key)
            return list(state["records"]), inserted

    def _with_records(self, body, protocol, records):
        if not records:
            return body
        key = "messages" if protocol == "claude" else "input"
        result = copy.deepcopy(body)
        updates = {index: effort for index, _, effort in records}
        items = []
        for index, item in enumerate(result[key]):
            if index in updates:
                items.append({"role": "system", "content": [], "output_config": {"effort": updates[index]}}
                             if protocol == "claude" else
                             {"type": "configuration_update", "reasoning": {"effort": updates[index]}})
            items.append(item)
        result[key] = items
        return result

    def _finish(self, body, protocol, key, items, index, version, records, decided, effort,
                log=None):
        final_records, inserted = self._commit(key, items, index, version, records, decided, effort)
        result = self._with_records(body, protocol, final_records)
        if inserted and log:
            note(log)
        elif final_records:
            note(f"harness={protocol} replayed={len(final_records)}")
        return result

    def rewrite(self, body, protocol="codex", threshold=None, capabilities=None, conversation_id=None):
        model = body.get("model", "")
        if protocol == "claude":
            levels = list(LEVELS) if model in CLAUDE_MODELS else []
            if body.get("thinking", {}).get("type") == "disabled":
                levels = [level for level in levels if level not in ("max", "xhigh")]
        else:
            if model not in CODEX_MODELS:
                note("unchanged: model capabilities unavailable")
                return body
            levels = (self.models if capabilities is None else capabilities).get(model, [])
        available = [level for level in LEVELS if level in levels]
        items_key = "messages" if protocol == "claude" else "input"
        items = body.get(items_key, [])
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            note("unchanged: unsupported input shape")
            return body
        if has_media(items):
            note("unchanged: non-text context")
            return body
        if not available or "low" not in available:
            note("unchanged: model capabilities unavailable")
            return body
        if any((protocol == "codex" and item.get("type") == "configuration_update") or
                (protocol == "claude" and item.get("role") == "system" and item.get("content", []) == [] and
                 "output_config" in item) for item in items):
            note("unchanged: per-message effort override in history")
            return body
        if protocol == "codex":
            reasoning = body.get("reasoning")
            extra_reasoning = set(reasoning or {}) - {"effort", "summary"} if isinstance(reasoning, dict) else set()
            # ponytail: Codex 0.153.2 emits context=all_turns metadata; other modes bypass routing.
            if (not isinstance(reasoning, (dict, type(None))) or
                    (extra_reasoning and not (extra_reasoning == {"context"} and
                                              reasoning.get("context") == "all_turns")) or
                    "context_management" in body or
                    ("truncation" in body and body["truncation"] != "disabled") or
                    ("previous_response_id" in body and body["previous_response_id"] is not None) or
                    any(item.get("type") == "compaction" for item in items)):
                note("unchanged: unsupported conversation mode")
                return body
        conversation_id = self._conversation_id(body, protocol, conversation_id)
        if not conversation_id or not items:
            note("unchanged: conversation key unavailable")
            return body
        insertion = claude_boundary(items) if protocol == "claude" else codex_boundary(items)
        try:
            key = self._state_key(protocol, model, conversation_id, items)
        except (TypeError, ValueError):
            note("unchanged: unsupported input shape")
            return body
        version, records, decided = self._state_snapshot(key, items)
        if insertion is None:
            result = self._with_records(body, protocol, records)
            if records:
                note(f"harness={protocol} replayed={len(records)}")
            return result
        anchor = item_hash(items[insertion])
        if (decided.get(insertion) == anchor or any(record[0] == insertion for record in records)):
            result = self._with_records(body, protocol, records)
            if records:
                note(f"harness={protocol} replayed={len(records)}")
            return result
        if len(records) >= RECORD_LIMIT:
            return self._finish(body, protocol, key, items, insertion, version, records, decided, None)
        state = claude_context(body) if protocol == "claude" else context_for(body)
        if not state:
            return self._finish(body, protocol, key, items, insertion, version, records, decided, None)
        previous = [record for record in records if record[0] < insertion]
        top = body.get("output_config" if protocol == "claude" else "reasoning")
        effective = previous[-1][2] if previous else (top.get("effort") if isinstance(top, dict) else None)
        start = time.monotonic()
        try:
            with self.lock:
                if self.pending is not None and not self.pending.done():
                    note("unchanged: classifier busy")
                    return self._finish(body, protocol, key, items, insertion, version, records, decided, None)
                self.pending = self.executor.submit(self.agent.predict, state, QUESTION)
                future = self.pending
            answer = future.result(timeout=self.timeout)["answers"]["effort"]
            confidence = float(answer["probabilities"][answer["choice"]])
            if answer["choice"] not in LEVELS or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("invalid prediction")
            if confidence < (self.threshold if threshold is None else threshold):
                note(f"unchanged: confidence={confidence:.2f}")
                return self._finish(body, protocol, key, items, insertion, version, records, decided, None)
            effort = next((level for level in available
                           if LEVELS.index(level) >= LEVELS.index(answer["choice"])), available[-1])
            return self._finish(body, protocol, key, items, insertion, version, records, decided,
                                effort if effort != effective else None,
                                f"harness={protocol} effort={effort} inserted_at={insertion} "
                                f"replayed={len(records)} probability={confidence:.2f} "
                                f"classifier_ms={(time.monotonic()-start)*1000:.0f}")
        except Exception as exc:
            # Exception messages may contain input data, so log only their class.
            note(f"unchanged: classifier {type(exc).__name__}")
            return self._finish(body, protocol, key, items, insertion, version, records, decided, None)

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
        rewrote = False
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
                conversation_id = self.headers.get("session-id") if protocol == "codex" else self.headers.get(
                    "X-Claude-Code-Session-Id")
                rewritten = self.server.router.rewrite(body, protocol, threshold, capabilities, conversation_id)
                if rewritten is not body:
                    rewrote = True
                    data = json.dumps(rewritten, ensure_ascii=False).encode()
            connection = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
            upstream = connection(target.hostname, target.port, timeout=600)
            headers = forwarded_headers(self.headers)
            if protocol == "claude" and rewrote:
                append_claude_beta(headers)
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
