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

## Claude 5.x hides its reasoning by default (`thinking.display="omitted"`)

**Re-diagnosed 2026-10-06 against genai.arizona.edu (was "zero reasoning on
Claude 5.x", misread 2026-09-17 as a reasoning defect).**

The 5.x line (and 4.7+) **does** reason — the earlier `reasoning_tokens=0` was not
a lack of reasoning, it was a *display* default. Anthropic's Messages API has a
`thinking.display` field; on Claude Opus 4.7+/4.8, the 5.x and 5.5 lines, and
fable/mythos it **defaults to `"omitted"`**: the model thinks, but the thinking
text is withheld, so `reasoning_tokens` reads 0 and `reasoning_content` is empty.
`Opus/Sonnet 4.6` and earlier default to `"summarized"` and return the text — which
is why the same adaptive request looked fine on 4.6.

| Request shape | claude-4-6-opus | claude-5-opus | claude-5-5-sonnet |
| --- | --- | --- | --- |
| `thinking: {type: adaptive}` | 200, real thinking | 200, `reasoning_tokens`=0, empty text | 200, `reasoning_tokens`=0, empty text |
| `thinking: {type: adaptive, display: summarized}` | 200, real thinking | 200, **real thinking** | 200, **real thinking** |
| adaptive + `display: summarized` + `output_config.effort: max` | n/a | 200, deeper thinking (effort honoured) | 200, real thinking |
| `reasoning_effort` alone | 200, real thinking (default display) | 200, effort honoured but text hidden | 200, effort honoured but text hidden |
| `reasoning_effort` + `thinking: {type: adaptive, display: summarized}` | n/a | 200, **real thinking** (proxy-injected shape verified 2026-10-06) | n/a |
| `thinking: {type: enabled, budget_tokens}` | 200, real thinking | 400 "Use adaptive and output_config.effort" | n/a |
| `thinking: {display: summarized}` (no type) | n/a | 400 `thinking.type: Field required` | n/a |
| `thinking: {type: adaptive, display: bogus}` | n/a | 400 `display: Input should be 'summarized', 'omitted'` | n/a |
| `thinking: {type: disabled, display: summarized}` | n/a | 400 `display: Extra inputs are not permitted` | n/a |

This is **not** a proxy translation bug: the injected body is exactly what was
sent, and `display="summarized"` restores the text on all four verified 5/5.5
opus/sonnet models, non-streaming and streaming alike. LiteLLM derives
`reasoning_tokens` from the thinking *text*, so a display-omitted block reports 0.

Signal pitfalls (carried over from the 2026-09-17 notes, now explained):

- `thinking_blocks` appears in **every** response — including requests with no
  `thinking` param at all — carrying a signature and an empty `thinking` string.
  That empty text is the `omitted` display, NOT evidence that no reasoning ran.
- The reliable signals for *visible* reasoning are
  `usage.completion_tokens_details.reasoning_tokens` and `reasoning_content`.

**Workaround (in this repo):** the thinking rewrite pass
(`_reconcile_thinking` in `src/open_webui/request_policy.py`) adds
`display="summarized"` for families that omit by default
(`ModelCapabilities.defaults_to_omitted_thinking`), and only where reasoning was
explicitly requested: it decorates an adaptive/enabled `thinking`, and synthesises
`{type: adaptive, display: summarized}` when a depth control (`reasoning_effort` or
`output_config.effort`) was supplied with no `thinking` key present.
A client's own `display` (including `"omitted"`) and a `type="disabled"`
are left untouched. 4.6 and earlier are unaffected (their default is already
visible). When it synthesises thinking from a depth control, it also raises `max_tokens` to the 64k floor the gateway requires for adaptive thinking.

**Removal trigger:** re-run the first table row on a 5.x/5.5 model. When adaptive
*without* `display` returns `reasoning_tokens > 0` (i.e. Anthropic flips the
default back to `"summarized"`), delete `_OMITTED_THINKING_MIN_VERSION`, the
`defaults_to_omitted_thinking` gate, and the visibility branch of
`_reconcile_thinking`.

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

## Rate limit is reported as a 400 with a detail string, not a 429

**Verified 2026-09-18 against genai.arizona.edu.**

The gateway enforces **two** request tiers per end user, and overage is
reported as HTTP **400**, not 429:

- 20 requests per 60 s for the end user, with a reset timestamp — the
  detail carries the exact reset time, so the proxy needs no window model
  for this tier:

```
{"detail":"Rate limit exceeded for end_user: josephmarbella@arizona.edu.
Limit type: requests. Current limit: 20, Remaining: 0. Limit resets at:
2026-09-18 15:54:15 UTC"}
```

- A burst tier of 10 requests per 60 s, whose detail carries **no reset
  timestamp** — only the limit and unit. The window boundary is not
  advertised; the proxy models it as a rolling 60 s window (ADR-0007):

```
{"detail":"Rate limit exceeded: 10 requests per minute. Please wait before
trying again."}
```

The sliding tier is per end user, **not per model** (the earlier "per-model"
text was wrong). Probes, 2026-09-18:

- A burst on one model exhausts the budget for every other model.
- `POST /api/chat/completions` and `POST /api/v1/chat/completions` share the
  same budget.
- Varying the `user` field in the request body (two alternate values)
  changed nothing — the limit keys off the authenticated end user.
- `GET /api/models` consumes no slot.
- A rejected request consumes no slot.

The sustainable rate is exactly 10/min: after a full burst of 10, no
request is admitted until the window frees.
No `Retry-After` and no `RateLimit-*` headers on either tier; where present,
the window-reset time exists only inside the detail string. A rejected request
does not consume a slot, and the end-user window reliably reopens at its
advertised reset.

**Workaround:** `is_rate_limit` / `RateLimitStall`
(`src/open_webui/rate_limit.py`) treat this body — and any 429 — as a rate
limit and stall pre-header requests (budget `RATE_LIMIT_STALL_MAX_SECONDS`,
default 300 s). The end-user tier stalls until the advertised reset; the
sliding tier stalls to the exact next free slot computed by
`SlidingWindowTracker` from this process's own admissions. On exhaustion the
proxy surfaces a 429 with `Retry-After` — for the sliding tier, the seconds
until the next slot. See [ADR-0006](adr/0006-rate-limit-stall.md) and
[ADR-0007](adr/0007-slot-aware-stall.md).
**Removal trigger:** the gateway starts emitting a 429 with `Retry-After`
(or `RateLimit-*` headers). Then drop the 400+detail sniff and keep the 429
path.
