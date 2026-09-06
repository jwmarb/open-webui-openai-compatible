"""Tests for the shared Open WebUI model-capability lookup."""

from __future__ import annotations

from src.open_webui.capabilities import (
    capabilities_for,
    normalize_model_id,
    split_thinking_suffix,
)


class TestSplitThinkingSuffix:
    def test_extended_suffix(self):
        assert split_thinking_suffix("claude-x:extended") == ("claude-x", ":extended")

    def test_adaptive_suffix(self):
        assert split_thinking_suffix("claude-x:adaptive") == ("claude-x", ":adaptive")

    def test_no_suffix(self):
        assert split_thinking_suffix("claude-x") == ("claude-x", None)


class TestNormalizeModelId:
    def test_separators_unified(self):
        assert normalize_model_id("anthropic.claude-x") == "anthropic-claude-x"
        assert normalize_model_id("bedrock_claude_x") == "bedrock-claude-x"
        assert normalize_model_id("vendor/claude-x") == "vendor-claude-x"

    def test_family_detection_survives_any_separator(self):
        for model in ("anthropic.claude-x", "bedrock_claude_x", "vendor/claude-x"):
            assert capabilities_for(model).is_anthropic is True

    def test_suffix_removed_before_normalizing(self):
        assert normalize_model_id("Claude-Opus:extended") == "claude-opus"


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


class TestSmallContext:
    def test_haiku_is_small_context(self):
        assert capabilities_for("claude-3-5-haiku").small_context is True

    def test_sonnet_is_not(self):
        assert capabilities_for("claude-4-6-sonnet").small_context is False


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


class TestBaseModel:
    def test_base_model_excludes_suffix(self):
        assert capabilities_for("claude-x:extended").base_model == "claude-x"
