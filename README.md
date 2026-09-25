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
  provider: ollama           # ollama | anthropic | claude-code | jev
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

When `provider: claude-code`, sup7 exposes two MCP tools (`sup7.pending`, `sup7.verdict`). Register sup7 as an MCP server in your mesh config — Claude Code pulls pending evaluations and submits verdicts. See [docs](https://docs.flux7.art/sup7/claude-code-callback/).

### Jev (TypeSafe AI)

Jev does not generate text: it answers typed questions about a state, each with a probability. sup7 asks four atomic questions about the pending call and combines them in code, fail-closed:

| Question | Type | Asks |
|----------|------|------|
| `decision` | choice | approve, escalate or deny |
| `destructive` | noul | deletes, overwrites or exfiltrates data, or changes permissions or secrets |
| `in_scope` | noul | consistent with the agent's recent activity |
| `injection` | noul | parameters carry instructions aimed at a model |

Approve only when `decision` is approve, `destructive` is low and `in_scope` is high (then `confidence_threshold` applies); deny only when deny is very probable; escalate everything else, including any API error. The probabilities are written into the decision reasoning, so each verdict is auditable in the mesh traces and in mem7.

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
├── logger.py           # JSONL decision log
└── providers/
    ├── base.py         # Provider interface
    ├── ollama.py       # Ollama HTTP provider
    ├── anthropic.py    # Claude Messages API
    ├── claude_code.py  # MCP callback provider
    └── jev.py          # TypeSafe Jev decision model (Cloudflare or TypeSafe)
```

## License

MIT

[docs.flux7.art/sup7](https://docs.flux7.art/sup7/) · [github.com/KTCrisis/flux7-supervisor](https://github.com/KTCrisis/flux7-supervisor)
