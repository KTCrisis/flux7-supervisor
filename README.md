# flux7-supervisor

Standalone L1 evaluation agent for [flux7-mesh](https://github.com/KTCrisis/flux7-mesh). Sits between the policy engine (L0) and human operators (L2) — polls pending approvals, evaluates them with rules and an LLM, and resolves or escalates.

```
flux7-mesh (L0: policy engine)
     │
     │  pending approval
     ▼
flux7-supervisor (L1: rules + LLM)
     │
     ├── rule match → auto-approve/deny
     ├── LLM evaluation → approve/deny/escalate
     └── unknown → escalate to human (L2)
```

## Install

```bash
pip install flux7-supervisor

# With Anthropic provider
pip install flux7-supervisor[anthropic]
```

## Quick start

```bash
# Check connectivity
sup7 -c sup7.yaml status

# Start the supervisor loop
sup7 -c sup7.yaml start
```

Requires a running `mesh7 serve` instance. Optional: `mem7 serve` for decision persistence.

## Run as a service

A systemd unit is provided in [`contrib/systemd/sup7.service`](contrib/systemd/sup7.service) (`After=`/`Wants=mesh7.service`, restart on failure):

```bash
sudo cp contrib/systemd/sup7.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now sup7
```

Adapt `User=` and paths first. Pair it with `approval.channel: queue` in the mesh config — a service has no TTY to prompt on.

## Configuration

```yaml
mesh:
  url: http://localhost:9090
  agent_id: supervisor

memory:
  url: http://localhost:9070
  enabled: true
  store_decisions: true

evaluator:
  provider: ollama           # ollama | anthropic | claude-code | jev | none (rules only)
  model: qwen3:14b
  url: http://localhost:11434
  timeout: 30
  confidence_threshold: 0.8

poll:
  interval: 2s

rules:
  - name: safe-reads
    condition: "tool contains read"
    action: approve
    confidence: 0.95
  - name: project-writes
    condition: "params.path starts_with project_dir"
    action: approve
    confidence: 0.9
  - name: injection-risk
    condition: "injection_risk == true"
    action: escalate
    confidence: 1.0

project_dirs:
  - /home/user/project
```

## Evaluation flow

1. **Poll** — fetches pending approvals from flux7-mesh
2. **Rules** — first-match-wins condition evaluation (instant)
3. **LLM** — if no rule matches, the configured provider evaluates with approval context
4. **Threshold** — if LLM confidence < `confidence_threshold`, escalate to human
5. **Resolve** — posts approve/deny back to flux7-mesh with reasoning
6. **Persist** — writes decision to flux7-memory as a queryable fact

## LLM providers

| Provider | Config | Use case |
|----------|--------|----------|
| `ollama` | Local HTTP, any model | Default. Fast, private, no API cost |
| `anthropic` | Claude Messages API | Higher quality evaluation, cloud |
| `claude-code` | MCP callback via `sup7.pending` + `sup7.verdict` tools | Claude Code acts as supervisor |
| `jev` | TypeSafe AI decision model, via Cloudflare Workers AI or the TypeSafe API | Fast typed decisions with probabilities, auditable |

### Claude Code callback

When `provider: claude-code`, sup7 is meant to expose two MCP tools (`sup7_pending`, `sup7_verdict`) that Claude Code pulls and answers. **Not functional yet:** the MCP server is defined (`mcp_server.py`) but not started by the runner, so use `ollama`, `anthropic` or `jev` meanwhile. See [docs](https://docs.flux7.art/sup7/claude-code-callback/).

### Provider chain

Several providers can be chained, so sup7 follows the evaluators a team actually has or is allowed to use, and survives a provider going down or away:

```yaml
evaluator:
  confidence_threshold: 0.8
  breaker_failures: 3       # consecutive failures before a provider is skipped
  breaker_cooldown: 300     # seconds it stays skipped
  chain:
    - provider: jev         # fast typed decisions when available
      jev: { backend: cloudflare, api_key_env: CLOUDFLARE_WORKERS_AI_TOKEN }
    - provider: ollama      # local, free, works offline
      model: qwen3:14b
```

Providers are tried in order; the first one that answers gives the verdict (an `escalate` verdict is an answer). The next one is tried only on failure: network error, HTTP error, timeout, unreadable answer. If every provider fails, sup7 escalates to a human. The reasoning records who decided, e.g. `[ollama, jev skipped] ...`. Without `chain`, the single `provider` works as before.

### Jev (TypeSafe AI)

Jev does not generate text: it answers typed questions about a state, each with a probability. sup7 asks six narrow factual questions (noul: probability of yes) about the pending call and decides in code, fail-closed:

| Question | Type | Asks |
|----------|------|------|
| `deletes` | noul | deletes files, directories or records |
| `overwrites` | noul | replaces an existing file or record (creating one does not count) |
| `exfiltrates` | noul | sends data off the machine |
| `secrets` | noul | reads, changes or exposes credentials, secrets or permissions |
| `in_scope` | noul | consistent with the agent's recent activity |
| `injection` | noul | parameters carry instructions aimed at a model |

`destructive` is the highest of the four harm signals. Injection above `injection_max` escalates; deny only when `destructive` ≥ `deny_min` and the call is out of scope; approve only when `destructive` ≤ `destructive_max` and `in_scope` ≥ `in_scope_min`, with confidence = the weakest safe-side signal (then `confidence_threshold` applies); escalate everything else, including any API error. The probabilities are written into the decision reasoning, so each verdict is auditable in the mesh traces and in mem7. Each Jev decision also records its provenance: the model version returned by the API (`jev-1.13.0`), a 12-character fingerprint of the question set (it changes whenever a question or criterion changes) and the thresholds applied, under `evaluator` in the decision log, mem7 and `GET /decisions`; model and fingerprint also head the reasoning (`Jev jev-1.13.0 q=…: approve (…)`), which is what the mesh trace keeps.

```yaml
evaluator:
  provider: jev
  confidence_threshold: 0.8
  jev:
    backend: cloudflare              # cloudflare (Workers AI, zero data retention) | typesafe
    api_key_env: CLOUDFLARE_API_TOKEN  # TYPESAFE_API_KEY with backend: typesafe
    account_id_env: CLOUDFLARE_ACCOUNT_ID
    destructive_max: 0.2
    in_scope_min: 0.7
    injection_max: 0.5
    deny_min: 0.9
    redact_params: [content]         # parameter names never sent to the model
```

## Admin API

Off by default. When enabled, sup7 serves a small HTTP API that flux7-console uses to show and steer the supervisor:

```yaml
admin:
  enabled: true
  host: 127.0.0.1   # loopback; set a token before binding elsewhere
  port: 9096
  token: ""         # when set, required as "Authorization: Bearer <token>"
```

| Route | Does |
|-------|------|
| `GET /health` | liveness (no token) |
| `GET /status` | running or paused, mesh reachability, decision counters, state of each provider (ok, failing, skipped by the breaker) |
| `GET /config` | rules, thresholds, provider chain; never secrets |
| `GET /decisions?limit=50` | most recent decisions with their reasoning |
| `POST /pause` | stop evaluating: approvals stay pending in the mesh, for a human |
| `POST /resume` | evaluate again |

## Rule conditions

```
"tool contains read"                       # tool name substring
"tool equals filesystem.read_file"         # exact match
"tool starts_with gmail"                   # prefix
"params.path starts_with project_dir"      # resolved against project_dirs list
"injection_risk == true"                   # boolean field
```

Operators: `contains`, `equals`, `starts_with`, `not_equals`, `==`, `!=`.

A catch-all escalation rule is auto-appended if not explicitly defined.

## How it fits

```
L0  flux7-mesh          Static policy (allow/deny/human_approval)    0ms
L1  flux7-mesh built-in  flux7-memory lookup (3+ past approvals)     ~100ms
L1+ flux7-supervisor     Rules + LLM evaluation                      ~2-20s
L2  Human                Claude Code prompt / flux7-console UI        minutes
```

The built-in L1 in flux7-mesh handles routine patterns. sup7 handles novel cases that need judgment. Both escalate unknowns to humans.

## Testing

```bash
pytest                  # 49 tests
pytest -x -v            # verbose, stop on first failure
```

## Project structure

```
src/sup7/
├── cli.py              # sup7 start / status
├── config.py           # YAML config loader
├── evaluator.py        # Orchestrates rules → LLM → resolve
├── rules.py            # Condition parser + predicate engine
├── runner.py           # Async poll loop with graceful shutdown
├── models.py           # Verdict, Decision, ApprovalContext
├── mcp_server.py       # FastMCP server for Claude Code callback
├── admin.py            # HTTP admin API (status, config, decisions, pause)
├── logger.py           # JSONL decision log
└── providers/
    ├── base.py         # Provider interface
    ├── ollama.py       # Ollama HTTP provider
    ├── anthropic.py    # Claude Messages API
    ├── claude_code.py  # MCP callback provider
    ├── jev.py          # TypeSafe Jev decision model (Cloudflare or TypeSafe)
    └── chain.py        # Provider chain with circuit breaker
```

## License

Apache 2.0, see [LICENSE](LICENSE).

[docs.flux7.art/sup7](https://docs.flux7.art/sup7/) · [github.com/KTCrisis/flux7-supervisor](https://github.com/KTCrisis/flux7-supervisor)
