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
`_strip_incompatible_reasoning_effort` drops all six fields.

**Consequence:** reasoning depth is **not controllable** for gpt-5.6 on this
gateway. A client asking for `xhigh` gets the default effort and a successful
response; the only signal is a WARNING in the proxy log.

**Removal trigger:** re-run the table above. When `reasoning_effort` returns 200
on a gpt-5.6 model, delete the family from `_REASONING_INCOMPATIBLE_RE` and drop
the pass.
