# Capability inference from model IDs

Open WebUI's `GET /api/models` returns `id`, `object`, `created` and `owned_by`.
It exposes no capability metadata, yet the gateway rejects requests whose
parameters do not match the model's family and version.

## Decision

Capability is inferred from the model ID in `src/open_webui/capabilities.py`,
and that module is the only place allowed to do so.

**The Anthropic family check is a positive allowlist** (`claude`, `fable`,
`mythos`), not "anything that is not OpenAI". An unrecognised model must never
receive a provider-specific parameter, because the failure mode is a hard 400
rather than a warning.

**IDs are normalised before matching** — lowercased with `.`, `_` and `/`
unified to `-` — so `anthropic.claude-x` and `bedrock_claude_x` agree.

**`supports_adaptive` and `requires_adaptive` are separate gates and must not be
collapsed** — even though only the `:adaptive` variant remains. `supports_adaptive`
gates variant generation; `requires_adaptive` coerces a *client-supplied*
`thinking.type="enabled"`, which arrives independently of any variant. Opus/Sonnet >= 4.6 *accept* `thinking.type="adaptive"`. Opus/Sonnet
>= 4.7 additionally *reject* `thinking.type="enabled"`. Claude 4.6 therefore
accepts both modes while 4.7+ accepts only adaptive. Verified against
genai.arizona.edu: `claude-4-6-opus` and `claude-4-6-sonnet` return 200 for
enabled thinking; `claude-5-opus` returns 400 demanding adaptive.

**Version digits are bounded to two** so a trailing date stamp
(`...-4-6-20250514`) is not read as a version.

**`accepts_effort_config` is a third, independent gate.** Claude 4.5 and earlier
reject `output_config.effort` with *"This model does not support the effort
parameter."*, 4.6 accepts the field and ignores it, and only 5.x acts on it. The
floor is therefore 4.6, not 5.0: accepting-and-ignoring is harmless, rejecting is
not. This does not line up with either adaptive gate — 4.6 accepts the effort
config while still permitting `thinking.type="enabled"` — so all three gates stay
separate comparisons. Verified 2026-09-06 against genai.arizona.edu.

## Consequences

- Both frontends resolve the `:adaptive` variant the same way.
- The `:extended` variant was removed, along with the thinking budgets and the
  Haiku `small_context` gate that existed only to size those budgets. Nothing
  else depended on them.
- A new Anthropic family is one entry in one frozenset.
- Three gates now derive from the same version tuple with three different
  thresholds (4.6 adaptive-capable, 4.7 adaptive-only, 4.6 effort-config). They
  are cheap to compare and expensive to conflate.
- The rules are empirical and will drift as the gateway changes. They belong in
  one module with tests, not spread across request-rewriting passes.
- Provider-specific quirks that are *upstream defects* rather than protocol
  rules go in `docs/upstream-compatibility.md` instead, with a removal trigger.
