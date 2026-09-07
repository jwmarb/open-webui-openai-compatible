# Upstream compatibility matrix

Workarounds for defects in the gateway, as opposed to protocol rules. Each entry
records what was verified, when, and what would let us delete the code.

Protocol-level capability rules live in
[ADR-0004](adr/0004-model-capability-inference.md).

## gpt-5.6 rejects every reasoning and verbosity control

**Verified 2026-09-06 against genai.arizona.edu.**

| Parameter | gpt-5.6-sol / terra / luna | claude-5-opus | gpt-oss-120b |
| --- | --- | --- | --- |
| `reasoning_effort` | 400 | 200 | 200 |
| `reasoning` | 400 | 200 | 200 |
| `effort` | 400 | 200 | 200 |
| `verbosity` | 400 | 200 | 200 |
| `textVerbosity` | 400 | 200 | 200 |
| `thinking` | 400 | 200 (Anthropic-only) | n/a |

The gpt-5.6 line is served via **Bedrock Converse**, which accepts none of
them. Three distinct upstream behaviours produce the failures:

- `reasoning_effort` is remapped by the gateway's LiteLLM onto Bedrock's
  Anthropic-only `thinking` parameter, so it fails with
  `400 unknown_parameter: 'thinking'` even though this proxy never sent
  `thinking`.
- `effort` and `textVerbosity` reach Bedrock verbatim and 400.
- `verbosity` trips `litellm.UnsupportedParamsError`.

**Workaround:** `capabilities_for()` reports
`accepts_reasoning_controls=False` for this family and
`_strip_incompatible_reasoning_effort` drops all seven fields (the six above
plus `output_config`).

**Consequence:** reasoning depth is **not controllable** for gpt-5.6 on this
gateway. A client asking for `xhigh` gets the default effort and a successful
response; the only signal is a WARNING in the proxy log.

**Removal trigger:** re-run the table above. When `reasoning_effort` returns 200
on a gpt-5.6 model, delete the family from `_REASONING_INCOMPATIBLE_RE` and drop
the pass.

## `output_config.effort` must be nested, and 4.5 rejects it outright

**Verified 2026-09-06 against genai.arizona.edu.**

Two separate rules, both empirical:

| Request shape | claude-4-5-haiku | claude-4-6-* | claude-5-* |
| --- | --- | --- | --- |
| `effort` (bare, top level) | 400 | 400 | 400 |
| `output_config: {effort}` | 400 | 200, no effect | 200, honoured |

- A **bare top-level `effort`** is rejected by every Bedrock model with
  `effort: Extra inputs are not permitted`. Only the nested form is accepted —
  upstream names it itself when refusing `thinking.type="enabled"`:
  *"Use `thinking.type.adaptive` and `output_config.effort` to control thinking
  behavior."*
- **Claude 4.5 and earlier** reject even the nested form with
  *"This model does not support the effort parameter."*
- **Claude 4.6** accepts the field and silently ignores it; only 5.x acts on it.
  The capability floor is therefore 4.6, since accepting-and-ignoring is
  harmless while rejecting is not.

**Workaround:** `translate_request` emits the nested `output_config`, and
`capabilities_for()` reports `accepts_effort_config=False` below 4.6 so
`_strip_incompatible_effort_config` removes just the `effort` key (a sibling
`format` key survives — it is not an effort control).

**Removal trigger:** when `output_config: {effort}` returns 200 on
`bedrock-claude-4-5-haiku`, delete `_EFFORT_CONFIG_MIN_VERSION` and the pass.

## Structured output is broken upstream for Claude

**Verified 2026-09-06 against genai.arizona.edu. Pre-existing; not worked around.**

The gateway's LiteLLM rewrites an OpenAI `response_format` into
`output_config.format`, which Bedrock then rejects:

```
400 output_config.format: Extra inputs are not permitted
```

This reproduces identically on **both** frontends — `POST /v1/chat/completions`
with a plain `response_format` and `POST /v1/messages` with
`output_config.format` fail the same way — so it is upstream, not a translation
defect. JSON-schema structured output is currently unavailable for Claude models
on this gateway.

No workaround is applied. Stripping `response_format` would silently discard an
explicit client contract and return free-form prose where the caller demanded
JSON; a 400 is the honest outcome.

**Removal trigger:** none needed in this repo. Retest after a gateway upgrade;
if it starts working, delete this section.
