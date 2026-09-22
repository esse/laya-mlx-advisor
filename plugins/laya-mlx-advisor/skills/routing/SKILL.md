---
name: routing
description: Set up or inspect Laya-MLX-Advisor's local reasoning-effort routing for Codex CLI and Claude Code. Use when the user asks about this plugin, its shared daemon, or its routing decisions.
---

Both local harnesses connect to one daemon loaded with laya-mlx. Routing happens
in the HTTP proxy before each supported inference request, not through skill
instructions or settings-file edits.

Find the source checkout via `LAYA_MLX_ADVISOR_ROOT` when launched through Laya; otherwise
use the user's known checkout path. Read its `README.md` for launch options.

- Run `bin/laya-mlx-advisor status` from the checkout to inspect the shared PID.
- Read `.runtime/daemon.log` for decisions and unchanged-request reasons.
- Start a new routed session with `bin/codex-laya` or `bin/claude-laya` in a terminal.
- Use `bin/laya-mlx-advisor stop` only when the user wants to stop routing; it affects both harnesses.

Installing this skill does not intercept an already-running session. Do not claim
that prompting for a different effort changes the API parameter. Do not launch an
interactive child harness inside an ongoing agent task merely to activate routing.

The implementation, virtualenv, checkpoint cache, logs, token, and locks all live
in the source checkout. Never start a second model worker or write effort settings
into either harness's configuration.
