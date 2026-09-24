# Laya-MLX-Advisor

Adaptive reasoning effort for **local Codex CLI and local Claude Code**, using one
resident [laya-mlx](https://github.com/mizorewww/laya-mlx) model on Apple Silicon.
Both harnesses send their model requests through the same local daemon. It
classifies each next step, including after tool results, and selects `low`,
`medium`, `high`, `xhigh`, or `max` according to its difficulty.

## Run

Requires Apple Silicon, macOS 14+, Python 3.11+, [uv](https://docs.astral.sh/uv/),
and an installed, authenticated Codex CLI or Claude Code.

```sh
./bin/setup

./bin/codex-laya
./bin/claude-laya
```

Run both at once: their startup messages report the **same daemon PID**. The first
launch downloads `aac6fef/laya-mlx` into this checkout and warms it up. Subsequent
launches reuse it. Only the daemon imports MLX; neither launcher loads a model.

Pass harness arguments after `--`:

```sh
./bin/codex-laya --target-model gpt-6-astra -- exec 'Investigate the failing tests'
./bin/claude-laya --target-model claude-opus-5-5 -- -p 'Investigate the failing tests'

# Codex with an API key instead of its existing ChatGPT login:
./bin/codex-laya --auth api -- --model gpt-6-astra

# Generic entrypoint, also usable from another working directory:
/absolute/path/to/this/repo/bin/laya-mlx-advisor claude -- -p 'Review this change'
```

`--auth api` requires `OPENAI_API_KEY`. Claude retains its existing Anthropic API
key or Claude subscription login; no credentials are copied or stored by this
project. Direct Anthropic service is supported; Bedrock, Vertex, Foundry, and
existing third-party gateways are not integrated.

### Use your usual `claude` and `codex` commands

Add these functions to your shell startup file (`~/.zshrc` for Zsh or
`~/.bashrc` for Bash), replacing the example path with this checkout's absolute
path:

```sh
claude() { "/absolute/path/to/Laya-MLX-Advisor/bin/claude-laya" -- "$@"; }
codex()  { "/absolute/path/to/Laya-MLX-Advisor/bin/codex-laya" -- "$@"; }
```

Reload that file or open a new terminal, then use `claude` and `codex` with your
usual arguments. The functions start or reuse the same daemon and configure
routing automatically. For Codex API-key authentication, add `--auth api` before
the function's `--` separator.

This applies to sessions launched through these shell functions; restart any
already-running agent sessions. Installing the plugin alone does not activate
routing. Use `command claude` or `command codex` to bypass the functions.

## Plugin installation

The repository contains both native plugin manifests in
`plugins/laya-mlx-advisor`, plus local marketplaces for both harnesses. Claude's
launcher loads this plugin for the session using `--plugin-dir`.

For persistent registration, run from this repository:

```sh
codex plugin marketplace add "$PWD"
codex plugin add laya-mlx-advisor@laya-mlx-advisor-local

claude plugin marketplace add "$PWD"
claude plugin install laya-mlx-advisor@laya-mlx-advisor-local
```

These optional commands write **the harnesses' own plugin registration/cache**.
The implementation and all Advisor-owned runtime files stay in this checkout.
Installing the plugin exposes its routing help skill; **start sessions through
the launchers to activate interception**. Plugin hooks cannot replace the
before-request proxy, and an already-running session cannot be retrofitted by
loading a skill. The checkout must remain available after registration.

## One process, one checkpoint

```text
Codex CLI ── /chatgpt or /openai ─┐
                                ├─ one local daemon + one Laya model ── respective provider
Claude Code ── /anthropic ────────┘
```

The daemon binds only `127.0.0.1` on an OS-assigned port. Its startup lock and
lifetime `flock` prevent simultaneous harness launches from loading duplicate
weights. A checkpoint mismatch is an error, not a reason to start another worker.
A stale registry is checked against a live, authenticated health endpoint.

- `.venv/`: shared Python dependencies.
- `.cache/`: uv and Hugging Face downloads.
- `.runtime/daemon.json`: PID, port, checkpoint, and local token; mode `0600`.
- `.runtime/daemon.log`: decisions and fallback reasons, without prompts or credentials.
- `.runtime/*.lock`: singleton coordination.

The daemon remains resident until explicitly stopped, including between CLI
sessions. Shutdown affects both harnesses; exit their routed sessions first.
Shutdown waits for any in-flight local inference before releasing the singleton lock.

```sh
./bin/laya-mlx-advisor status
tail -f .runtime/daemon.log
./bin/laya-mlx-advisor stop
```

## Routing and limits

Laya sees short, bounded excerpts of the latest user task and recent tool/assistant
messages, not the entire transcript or encrypted reasoning. This is a heuristic,
not a trained or benchmarked coding-difficulty classifier. Validate the quality
tradeoff on your own tasks before relying on savings.

The default decision threshold is **0.70 winning-class probability**. Laya's
separate entropy-derived `confidence` field is deliberately not used. Tune it per
launcher, for example `./bin/claude-laya --threshold 0.85`. Below the threshold,
on an error, while inference is busy, or after a two-second deadline, the original
request is forwarded unchanged. Routing decisions appear in the daemon log.

The five choices describe mechanical work (`low`), routine implementation
(`medium`), multistep debugging (`high`), complex interactions (`xhigh`), and
exceptionally difficult research or proofs (`max`). If a model lacks the selected
level, routing uses the next higher supported level, capped at its maximum.
The threshold still applies to the winning choice; with five choices it may
preserve the original effort more often than the former two-choice classifier.

Codex capabilities come from its existing `models_cache.json`. Unknown models
pass through. To supply known capabilities explicitly:

```sh
./bin/codex-laya --target-model gpt-6-astra --efforts low,medium,high,xhigh,max
```

Neither harness is upgraded to `ultra`; Claude requests with thinking explicitly
disabled are capped at `high`. Token-counting and compaction endpoints are
forwarded without classification.

Cache-safe routing is enabled only for Claude `claude-fable-5-1`,
`claude-mythos-5-1`, `claude-opus-5-5`, and `claude-opus-5`, plus Codex
`gpt-6-astra`, `gpt-6-sol`, and `gpt-6-luna` when their catalog or explicit
capabilities list includes `low`. For these models, the proxy keeps the
harness-authored top-level effort unchanged and inserts per-message effort
updates before new user/tool-result items, then replays those updates at their
original positions on later requests. Unsupported models, media, provider
compaction modes, and existing per-message overrides pass through untouched.

Routing state is bounded in memory, so a daemon restart, LRU eviction, or
history rewrite may cause cache misses. A classifier timeout, error, or low
confidence replays existing updates without creating a new one. Claude's
`count_tokens` endpoint is forwarded without rewriting and may slightly
undercount the injected items. This follows Anthropic's
[per-message effort](https://platform.claude.com/docs/en/build-with-claude/effort)
and OpenAI's
[mid-conversation reasoning updates](https://developers.openai.com/api/docs/guides/reasoning?api-mode=responses#change-reasoning-mid-conversation)
requirements.

Codex uses HTTP streaming instead of WebSockets so every model request passes
through routing. HTTP errors and streamed response bytes are relayed;
credentials stay in memory and are forwarded only to fixed provider endpoints.

The request formats and launch controls were checked against Codex CLI **0.153.2**
and Claude Code **2.1.281**. Claude subscription routing uses the installed CLI's
`_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL` compatibility flag. Fake-upstream tests
exercise both real CLIs, not subscription billing or live provider acceptance.

## Verify

```sh
python3 -m unittest -v
python3 scripts/check_singleton.py
```

The standard tests use local fake upstreams and no model downloads or paid API
calls. The CLI integration test needs both harness executables. The singleton
check starts the **real** local checkpoint, launches both harnesses in version-only
mode, asserts one PID, and checks that a second daemon cannot acquire the lock.
It leaves the daemon available for use; stop it with the command above.

Sources: [Codex hooks](https://learn.chatgpt.com/docs/hooks),
[Codex providers](https://learn.chatgpt.com/docs/config-file/config-advanced),
[Claude effort](https://platform.claude.com/docs/en/build-with-claude/effort), and
[Claude gateway authentication](https://code.claude.com/docs/en/llm-gateway).
