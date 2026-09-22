#!/usr/bin/env python3
"""One repository-local Laya process, serving both harness proxies."""
import argparse
import fcntl
import http.client
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = "aac6fef/laya-mlx"
PROTOCOL = 3


def state_dir():
    return ROOT / ".runtime"


class Client:
    def __init__(self, info, timeout=2):
        self.info, self.timeout = info, timeout

    def request(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", self.info["port"], timeout=self.timeout)
        try:
            conn.request("POST" if path == "/shutdown" else "GET", path,
                         headers={"X-Laya-Advisor-Token": self.info["token"]})
            response = conn.getresponse()
            data = response.read(65536)
            if response.status != 200:
                raise RuntimeError(f"Laya service HTTP {response.status}")
            return json.loads(data)
        finally:
            conn.close()

    @classmethod
    def existing(cls, model=MODEL):
        try:
            info = json.loads((state_dir() / "daemon.json").read_text())
            client = cls(info)
            health = client.request("/health")
            if health.get("pid") != info["pid"]:
                return None
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, http.client.HTTPException):
            return None
        if health.get("protocol") != PROTOCOL:
            raise RuntimeError("A different daemon version is running; stop it before upgrading")
        if health.get("model") != model:
            raise RuntimeError(f"Shared Laya already serves {health.get('model')}; stop it before changing checkpoints")
        return client

    @classmethod
    def connect(cls, model=MODEL, startup_timeout=120):
        root = state_dir()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        deadline = time.monotonic() + startup_timeout
        # Startup is serialized across both harnesses, even when they start simultaneously.
        with (root / "start.lock").open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Waiting for shared Laya startup")
                    time.sleep(0.1)
            client = cls.existing(model)
            if client:
                return client
            python = ROOT / ".venv/bin/python"
            if not python.is_file():
                raise RuntimeError(f"Run {ROOT / 'bin/setup'} first")
            # The model and all dependency/download caches live inside this repository.
            env = {**os.environ, "HF_HOME": str(ROOT / ".cache/huggingface"),
                   "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
            with (root / "daemon.log").open("ab") as log:
                process = subprocess.Popen([str(python), str(Path(__file__).resolve()), "--serve", "--model", model],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, start_new_session=True)
            while time.monotonic() < deadline:
                client = cls.existing(model)
                if client:
                    return client
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"Shared Laya startup failed; see {root / 'daemon.log'}")
                # A zero exit can mean another starter already owns the lifetime lock.
                time.sleep(0.1)
            raise TimeoutError(f"Shared Laya startup timed out; see {root / 'daemon.log'}")


def serve(model):
    root = state_dir()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "daemon.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return  # Crucially, do not import MLX or load weights before winning this lock.
        import laya_mlx
        from laya_mlx_advisor import Proxy, Router, QUESTION
        agent = laya_mlx.load(model)
        agent.predict("Run a known test command.", QUESTION)
        router = Router(agent, {})
        server = Proxy(router, None, secrets.token_hex(32))
        server.info = {"protocol": PROTOCOL, "pid": os.getpid(), "port": server.server_port,
                       "model": model, "token": server.token, "root": str(ROOT)}
        temporary = root / f"daemon.{os.getpid()}.tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(server.info, stream)
        temporary.replace(root / "daemon.json")
        signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
        try:
            server.serve_forever()
        finally:
            server.server_close()
            router.close()
            (root / "daemon.json").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    if args.serve:
        serve(args.model)
    else:
        client = Client.connect(args.model)
        print(json.dumps(client.request("/health")))


if __name__ == "__main__":
    main()
