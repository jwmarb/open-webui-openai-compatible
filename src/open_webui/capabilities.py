"""What the Open WebUI gateway will accept for a given model ID.

One capability lookup shared by both frontends. Open WebUI's `GET /api/models`
exposes no capability metadata, so the model ID is the only available signal
and every rule here is empirical — verified against the live gateway and
recorded in `docs/adr/0004-model-capability-inference.md`.

`supports_adaptive` gates the `:adaptive` thinking variant. `requires_adaptive`
is separate and still needed even without variants: Claude 4.7+ rejects a
client-supplied `thinking.type="enabled"`, which the request policy coerces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = [
    "ModelCapabilities",
    "THINKING_SUFFIX_ADAPTIVE",
    "capabilities_for",
    "normalize_model_id",
    "split_thinking_suffix",
]

THINKING_SUFFIX_ADAPTIVE: Final[str] = ":adaptive"

# Allowlist, not "not OpenAI": an unrecognised model must never receive a
# provider-specific parameter. Add new Anthropic families here as they ship.
_ANTHROPIC_FAMILY_TOKENS: Final[frozenset[str]] = frozenset({"claude", "fable", "mythos"})

_ADAPTIVE_MIN_VERSION: Final[tuple[int, int]] = (4, 6)
_ADAPTIVE_ONLY_MIN_VERSION: Final[tuple[int, int]] = (4, 7)
_ADAPTIVE_CAPABLE_LINES: Final[frozenset[str]] = frozenset({"opus", "sonnet"})
_ADAPTIVE_ALWAYS_CAPABLE: Final[frozenset[str]] = frozenset({"fable", "mythos"})

# Bounded to two digits so trailing date stamps ("...-4-6-20250514") are not
# misread as versions.
_VERSION_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<!\d)(\d{1,2})(?:-(\d{1,2}))?(?!\d)")

# The gpt-5.6 line is served via Bedrock Converse, which accepts no reasoning or
# verbosity controls at all. See docs/upstream-compatibility.md.
_REASONING_INCOMPATIBLE_RE: Final[re.Pattern[str]] = re.compile(r"gpt-5-6")


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    base_model: str
    is_anthropic: bool
    supports_adaptive: bool
    requires_adaptive: bool
    accepts_reasoning_controls: bool

    @property
    def supports_thinking(self) -> bool:
        return self.is_anthropic


def split_thinking_suffix(model: str) -> tuple[str, str | None]:
    if model.endswith(THINKING_SUFFIX_ADAPTIVE):
        return model[: -len(THINKING_SUFFIX_ADAPTIVE)], THINKING_SUFFIX_ADAPTIVE
    return model, None


def normalize_model_id(model_id: str) -> str:
    base, _ = split_thinking_suffix(model_id)
    return re.sub(r"[._/]", "-", base.lower())


def _adaptive_version(normalized: str, is_anthropic: bool) -> tuple[int, int]:
    if not is_anthropic:
        return (0, 0)
    if any(token in normalized for token in _ADAPTIVE_ALWAYS_CAPABLE):
        return (99, 99)
    if not any(line in normalized for line in _ADAPTIVE_CAPABLE_LINES):
        return (0, 0)
    matches = _VERSION_PATTERN.findall(normalized)
    if not matches:
        return (0, 0)
    major, minor = matches[0]
    return int(major), int(minor or 0)


def capabilities_for(model_id: str) -> ModelCapabilities:
    base, _ = split_thinking_suffix(model_id)
    normalized = normalize_model_id(model_id)
    is_anthropic = any(token in normalized for token in _ANTHROPIC_FAMILY_TOKENS)
    version = _adaptive_version(normalized, is_anthropic)

    return ModelCapabilities(
        base_model=base,
        is_anthropic=is_anthropic,
        supports_adaptive=version >= _ADAPTIVE_MIN_VERSION,
        requires_adaptive=version >= _ADAPTIVE_ONLY_MIN_VERSION,
        accepts_reasoning_controls=not _REASONING_INCOMPATIBLE_RE.search(normalized),
    )
