"""Tests for the shared Open WebUI model-capability lookup."""

from __future__ import annotations

from src.open_webui.capabilities import (
    capabilities_for,
    normalize_model_id,
    split_thinking_suffix,
)


class TestSplitThinkingSuffix:
    def test_adaptive_suffix(self):
        assert split_thinking_suffix("claude-x:adaptive") == ("claude-x", ":adaptive")

    def test_no_suffix(self):
        assert split_thinking_suffix("claude-x") == ("claude-x", None)

    def test_extended_suffix_is_no_longer_recognised(self):
        assert split_thinking_suffix("claude-x:extended") == ("claude-x:extended", None)


class TestNormalizeModelId:
    def test_separators_unified(self):
        assert normalize_model_id("anthropic.claude-x") == "anthropic-claude-x"
        assert normalize_model_id("bedrock_claude_x") == "bedrock-claude-x"
        assert normalize_model_id("vendor/claude-x") == "vendor-claude-x"

    def test_family_detection_survives_any_separator(self):
        for model in ("anthropic.claude-x", "bedrock_claude_x", "vendor/claude-x"):
            assert capabilities_for(model).is_anthropic is True

    def test_suffix_removed_before_normalizing(self):
        assert normalize_model_id("Claude-Opus:adaptive") == "claude-opus"


class TestAnthropicFamilyDetection:
    def test_claude_is_anthropic(self):
        assert capabilities_for("claude-sonnet-4-20250514").is_anthropic is True

    def test_fable_is_anthropic(self):
        assert capabilities_for("fable-1").is_anthropic is True

    def test_mythos_is_anthropic(self):
        assert capabilities_for("mythos-2").is_anthropic is True

    def test_gpt_is_not_anthropic(self):
        assert capabilities_for("gpt-4o").is_anthropic is False

    def test_unknown_model_is_not_anthropic(self):
        assert capabilities_for("llama3").is_anthropic is False


class TestAdaptiveGates:
    def test_claude_4_5_supports_neither(self):
        caps = capabilities_for("claude-4-5-sonnet")
        assert caps.supports_adaptive is False
        assert caps.requires_adaptive is False

    def test_claude_4_6_supports_but_does_not_require(self):
        caps = capabilities_for("claude-4-6-opus")
        assert caps.supports_adaptive is True
        assert caps.requires_adaptive is False

    def test_claude_4_7_requires_adaptive(self):
        caps = capabilities_for("claude-4-7-opus")
        assert caps.supports_adaptive is True
        assert caps.requires_adaptive is True

    def test_claude_5_requires_adaptive(self):
        caps = capabilities_for("claude-5-opus")
        assert caps.supports_adaptive is True
        assert caps.requires_adaptive is True

    def test_fable_requires_adaptive(self):
        caps = capabilities_for("fable-1")
        assert caps.supports_adaptive is True
        assert caps.requires_adaptive is True

    def test_haiku_supports_neither(self):
        caps = capabilities_for("claude-3-5-haiku")
        assert caps.supports_adaptive is False
        assert caps.requires_adaptive is False

    def test_trailing_date_stamp_is_not_read_as_version(self):
        caps = capabilities_for("claude-sonnet-4-6-20250514")
        assert caps.supports_adaptive is True
        assert caps.requires_adaptive is False


class TestReasoningControls:
    def test_gpt_5_6_rejects_reasoning_controls(self):
        assert capabilities_for("gpt-5.6-sol").accepts_reasoning_controls is False

    def test_gpt_5_6_variants_all_rejected(self):
        for model in ("gpt-5-6-terra", "gpt-5.6-luna", "gpt_5_6_sol"):
            assert capabilities_for(model).accepts_reasoning_controls is False

    def test_gpt_oss_accepts(self):
        assert capabilities_for("gpt-oss-120b").accepts_reasoning_controls is True

    def test_claude_accepts(self):
        assert capabilities_for("claude-5-opus").accepts_reasoning_controls is True


class TestEffortConfigGate:
    def test_claude_4_5_rejects_effort_config(self):
        assert capabilities_for("bedrock-claude-4-5-haiku").accepts_effort_config is False

    def test_claude_3_rejects_effort_config(self):
        assert capabilities_for("claude-3-5-sonnet").accepts_effort_config is False

    def test_claude_4_6_accepts_effort_config(self):
        assert capabilities_for("bedrock-claude-4-6-sonnet").accepts_effort_config is True

    def test_claude_5_accepts_effort_config(self):
        assert capabilities_for("bedrock-claude-5-opus").accepts_effort_config is True

    def test_non_anthropic_models_are_not_gated(self):
        for model in ("gpt-oss-120b", "google.gemma-4-31b", "bedrock-nova-pro-v1"):
            assert capabilities_for(model).accepts_effort_config is True, model

    def test_effort_gate_is_independent_of_adaptive_gates(self):
        """4.6 accepts the field yet does not require adaptive thinking, so the
        three gates cannot be collapsed into one comparison.
        """
        caps = capabilities_for("bedrock-claude-4-6-sonnet")
        assert caps.accepts_effort_config is True
        assert caps.supports_adaptive is True
        assert caps.requires_adaptive is False


class TestThinkingDisplayGate:
    """``defaults_to_omitted_thinking`` — the fourth, independent gate.

    Claude 4.7+/5.x (and fable/mythos) hide their thinking text by default
    (``thinking.display`` defaults to ``"omitted"``); 4.6 and earlier do not.
    """

    def test_claude_4_5_does_not_omit(self):
        assert capabilities_for("bedrock-claude-4-5-haiku").defaults_to_omitted_thinking is False

    def test_claude_4_6_does_not_omit(self):
        assert capabilities_for("bedrock-claude-4-6-opus").defaults_to_omitted_thinking is False

    def test_claude_4_7_omits(self):
        assert capabilities_for("bedrock-claude-4-7-opus").defaults_to_omitted_thinking is True

    def test_claude_5_omits(self):
        assert capabilities_for("bedrock-claude-5-opus").defaults_to_omitted_thinking is True

    def test_claude_5_5_omits(self):
        assert capabilities_for("bedrock-claude-5-5-sonnet").defaults_to_omitted_thinking is True

    def test_fable_omits(self):
        assert capabilities_for("fable-5").defaults_to_omitted_thinking is True

    def test_mythos_omits(self):
        assert capabilities_for("mythos-5").defaults_to_omitted_thinking is True

    def test_non_anthropic_does_not_omit(self):
        for model in ("openai.gpt-5.6-luna", "gpt-oss-120b", "meta.llama3"):
            assert capabilities_for(model).defaults_to_omitted_thinking is False, model

    def test_display_gate_is_independent_of_adaptive_gates(self, monkeypatch):
        """The display gate computes from its own floor constant, not from
        ``requires_adaptive``. Monkeypatching the omitted-floor to 5.0
        decouples the two gates on 4.7 models: ``requires_adaptive`` stays
        True (4.7 >= 4.7) while ``defaults_to_omitted_thinking`` flips to
        False (4.7 < 5.0).
        """
        import src.open_webui.capabilities as caps_module

        caps_46 = capabilities_for("bedrock-claude-4-6-opus")
        assert caps_46.requires_adaptive is False
        assert caps_46.defaults_to_omitted_thinking is False

        caps_47 = capabilities_for("bedrock-claude-4-7-opus")
        assert caps_47.requires_adaptive is True
        assert caps_47.defaults_to_omitted_thinking is True

        # Decouple the floors to prove the gates are independently computed.
        monkeypatch.setattr(caps_module, "_OMITTED_THINKING_MIN_VERSION", (5, 0))
        caps_47_decoupled = capabilities_for("bedrock-claude-4-7-opus")
        assert caps_47_decoupled.requires_adaptive is True
        assert caps_47_decoupled.defaults_to_omitted_thinking is False


class TestBaseModel:
    def test_base_model_excludes_suffix(self):
        assert capabilities_for("claude-x:adaptive").base_model == "claude-x"
