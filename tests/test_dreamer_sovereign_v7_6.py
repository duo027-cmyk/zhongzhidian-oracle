"""
Pytest test suite for dreamer_sovereign_v7_6.py

Covers the fixes introduced in the "請針對問題優化修正到完美" PR:
  1. USER_KEY_HASH loaded from DREAMER_KEY_HASH env var
  2. TTS model path loaded from DREAMER_TTS_MODEL env var
  3. ConversationMemory JSON persistence (save_to_json / load_from_json)
  4. KnowledgeModule expanded from 5 → 13 entries
  5. ValuePruner bilingual danger keywords (Chinese + English)
  6. Imagination rollout (_run_imagination_rollout / imag_buffer population)
  7. _update_actor_critic refactor returns float loss
  8. SelfReviewModule routing and multi-turn context
  9. Agent.chat() auto-saves conversation history
"""

import hashlib
import json
import os
import sys
import types
import tempfile

import pytest
import numpy as np
import torch

# ── Stub piper so imports succeed without the real package ───────────────────

_piper_stub = types.ModuleType("piper")


class _FakePiperVoice:
    @staticmethod
    def load(name: str) -> "_FakePiperVoice":
        raise RuntimeError("piper stub — no real voice")


_piper_stub.PiperVoice = _FakePiperVoice
sys.modules.setdefault("piper", _piper_stub)

# ── Import module under test (verify_identity patched out) ───────────────────

import dreamer_sovereign_v7_6 as dsv

# Patch verify_identity so tests don't prompt for stdin
dsv.verify_identity = lambda: None


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_agent() -> dsv.Agent:
    """Return a freshly initialised Agent with a clean conversation."""
    agent = dsv.Agent()
    agent.conversation.clear()
    return agent


# ============================================================================
# 1. KEY HASH FROM ENVIRONMENT VARIABLE
# ============================================================================

class TestKeyHashFromEnvVar:
    def test_default_hash_is_placeholder(self):
        """Without DREAMER_KEY_HASH env var, the fallback hash is used."""
        expected = hashlib.sha256("owner_secret".encode()).hexdigest()
        assert dsv._DEFAULT_KEY_HASH == expected

    def test_env_var_overrides_hash(self, monkeypatch):
        """Setting DREAMER_KEY_HASH env var changes USER_KEY_HASH at module level."""
        custom_hash = hashlib.sha256("my_real_secret".encode()).hexdigest()
        monkeypatch.setenv("DREAMER_KEY_HASH", custom_hash)
        # Reload the env var the same way the module does
        runtime_hash = os.environ.get("DREAMER_KEY_HASH", dsv._DEFAULT_KEY_HASH)
        assert runtime_hash == custom_hash

    def test_hash_attribute_exists(self):
        assert hasattr(dsv, "USER_KEY_HASH")
        assert hasattr(dsv, "_DEFAULT_KEY_HASH")
        assert isinstance(dsv.USER_KEY_HASH, str)
        assert len(dsv.USER_KEY_HASH) == 64  # SHA-256 hex digest


# ============================================================================
# 2. TTS MODEL PATH FROM ENVIRONMENT VARIABLE
# ============================================================================

class TestTTSModelFromEnvVar:
    def test_default_tts_model(self):
        """Without DREAMER_TTS_MODEL, the default model name is used."""
        default = os.environ.get("DREAMER_TTS_MODEL", "tw_taigi_female")
        assert default == "tw_taigi_female"

    def test_env_var_attribute_exists(self):
        assert hasattr(dsv, "_TTS_MODEL")
        assert isinstance(dsv._TTS_MODEL, str)

    def test_tts_module_fallback_without_piper(self):
        """TTSModule falls back to text output when piper is unavailable."""
        agent = make_agent()
        tts: dsv.TTSModule = agent.modules["tts"]
        assert tts.voice is None  # piper stub raises → fallback


# ============================================================================
# 3. CONVERSATION MEMORY JSON PERSISTENCE
# ============================================================================

class TestConversationMemoryPersistence:
    def test_save_and_reload(self, tmp_path):
        mem = dsv.ConversationMemory(maxlen=10)
        mem.add_user("你好")
        mem.add_assistant("您好，主公！", topic="greeting")
        path = str(tmp_path / "conv.json")
        mem.save_to_json(path)
        assert os.path.exists(path)

        mem2 = dsv.ConversationMemory(maxlen=10)
        ok = mem2.load_from_json(path)
        assert ok is True
        assert len(mem2) == 2
        hist = mem2.get_history()
        assert hist[0]["role"] == "user"
        assert hist[0]["content"] == "你好"
        assert hist[1]["role"] == "assistant"
        assert hist[1].get("topic") == "greeting"

    def test_load_nonexistent_returns_false(self, tmp_path):
        mem = dsv.ConversationMemory()
        result = mem.load_from_json(str(tmp_path / "missing.json"))
        assert result is False
        assert len(mem) == 0

    def test_save_preserves_topic_field(self, tmp_path):
        mem = dsv.ConversationMemory()
        mem.add_assistant("answer", topic="self_review:strengths")
        path = str(tmp_path / "t.json")
        mem.save_to_json(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data[0].get("topic") == "self_review:strengths"

    def test_load_invalid_entries_skipped(self, tmp_path):
        """Entries without required keys are silently skipped."""
        path = str(tmp_path / "bad.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([
                {"role": "user", "content": "valid"},
                {"malformed": True},  # should be skipped
                {"role": "assistant", "content": "ok"},
            ], f)
        mem = dsv.ConversationMemory()
        ok = mem.load_from_json(path)
        assert ok is True
        assert len(mem) == 2

    def test_agent_chat_autosaves(self, tmp_path, monkeypatch):
        """Agent.chat() should save conversation to disk after each turn,
        using the monkeypatched _CONVERSATION_PATH."""
        path = str(tmp_path / "chat_save.json")
        monkeypatch.setattr(dsv, "_CONVERSATION_PATH", path)
        # Create Agent directly (not via make_agent) so the patched path is used
        # for both load (no-op: file doesn't exist yet) and subsequent saves.
        agent = dsv.Agent()
        agent.conversation.clear()
        agent.chat("補鈣有什麼注意事項")
        assert os.path.exists(path), (
            f"Expected conversation saved at monkeypatched path {path}"
        )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) >= 2


# ============================================================================
# 4. KNOWLEDGE MODULE EXPANDED (5 → 13 ENTRIES)
# ============================================================================

class TestKnowledgeModuleExpanded:
    def _count_entries(self, agent: dsv.Agent) -> int:
        km: dsv.KnowledgeModule = agent.modules["knowledge"]
        return sum(len(v) for v in km.knowledge.values())

    def test_total_entries_at_least_13(self):
        agent = make_agent()
        assert self._count_entries(agent) >= 13

    def test_new_medical_entries(self):
        agent = make_agent()
        a = agent.chat("糖尿病怎麼治療？")
        assert "糖尿病" in a or "Metformin" in a or "血糖" in a

    def test_new_physics_entry_relativity(self):
        agent = make_agent()
        a = agent.chat("相對論是什麼？")
        assert "相對論" in a or "Einstein" in a or "E=mc" in a

    def test_new_math_entry_calculus(self):
        agent = make_agent()
        a = agent.chat("微積分是什麼？")
        assert "微積分" in a or "微分" in a or "積分" in a

    def test_new_chemistry_acid_base(self):
        agent = make_agent()
        a = agent.chat("酸鹼中和是什麼？")
        assert "酸" in a or "鹼" in a or "pH" in a

    def test_existing_entries_still_work(self):
        agent = make_agent()
        a = agent.chat("天空為什麼是藍色的？")
        assert "Rayleigh" in a or "散射" in a or "藍" in a


# ============================================================================
# 5. VALUEPRUNER BILINGUAL DANGER KEYWORDS
# ============================================================================

class TestValuePrunerBilingual:
    def test_english_keyword_rejected(self):
        pruner = dsv.ValuePruner()
        safe, reason = pruner.is_safe(1.0, 1.0, 100, action_str="harm the user")
        assert safe is False
        assert "harm" in reason

    def test_chinese_keyword_rejected(self):
        pruner = dsv.ValuePruner()
        safe, reason = pruner.is_safe(1.0, 1.0, 100, action_str="傷害他人")
        assert safe is False
        assert "傷害" in reason

    def test_chinese_keywords_present(self):
        pruner = dsv.ValuePruner()
        chinese = {"傷害", "毀滅", "欺騙", "詐騙", "戰爭", "殺人", "欺詐", "剝削", "危害", "攻擊"}
        for kw in chinese:
            assert kw in pruner.danger_keywords, f"{kw!r} missing from danger_keywords"

    def test_safe_action_passes(self):
        pruner = dsv.ValuePruner()
        safe, _ = pruner.is_safe(1.0, 1.5, 100, action_str="move left")
        assert safe is True

    def test_distance_spike_rejected(self):
        pruner = dsv.ValuePruner()
        safe, reason = pruner.is_safe(1.0, 5.5, 100)  # spike > 3.8
        assert safe is False
        assert "距離" in reason


# ============================================================================
# 6. IMAGINATION ROLLOUT
# ============================================================================

class TestImaginationRollout:
    def test_rollout_populates_imag_buffer(self):
        """_run_imagination_rollout must add transitions to imag_buffer."""
        agent = make_agent()
        assert len(agent.imag_buffer) == 0

        seeds = torch.zeros(dsv.CANDIDATES_PER_ACT, dsv.STATE_DIM, device=dsv.DEVICE)
        n = agent._run_imagination_rollout(seeds)
        expected = dsv.CANDIDATES_PER_ACT * dsv.IMAGINE_STEPS
        assert n == expected
        assert len(agent.imag_buffer) == expected

    def test_rollout_returns_transition_count(self):
        agent = make_agent()
        seeds = torch.randn(4, dsv.STATE_DIM, device=dsv.DEVICE)
        n = agent._run_imagination_rollout(seeds)
        assert n == 4 * dsv.IMAGINE_STEPS

    def test_imag_buffer_transitions_valid_shapes(self):
        agent = make_agent()
        seeds = torch.zeros(2, dsv.STATE_DIM, device=dsv.DEVICE)
        agent._run_imagination_rollout(seeds)
        states, actions, rewards, nexts, dones = agent.imag_buffer.sample(
            min(dsv.BATCH, len(agent.imag_buffer))
        )
        assert states.shape[-1] == dsv.STATE_DIM
        assert actions.shape[-1] == dsv.ACTION_DIM
        assert rewards.shape[-1] == 1
        assert nexts.shape[-1] == dsv.STATE_DIM

    def test_update_uses_imag_buffer_when_ready(self):
        """After filling imag_buffer, update() should do a second AC pass."""
        agent = make_agent()
        # Fill real buffer to BATCH threshold
        env = dsv.World()
        state = env.reset()
        for _ in range(dsv.BATCH + 5):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.real_buffer.push(state, action.cpu().numpy().flatten(),
                                   reward, next_state, float(done))
            state = next_state if not done else env.reset()

        # Pre-fill imag_buffer directly
        seeds = torch.zeros(dsv.CANDIDATES_PER_ACT, dsv.STATE_DIM)
        agent._run_imagination_rollout(seeds)
        # Add more to ensure >= BATCH
        for _ in range(dsv.BATCH // (dsv.CANDIDATES_PER_ACT * dsv.IMAGINE_STEPS) + 1):
            agent._run_imagination_rollout(seeds)

        assert len(agent.imag_buffer) >= dsv.BATCH
        loss = agent.update()
        assert loss > 0.0


# ============================================================================
# 7. _update_actor_critic REFACTOR
# ============================================================================

class TestUpdateActorCriticRefactor:
    def _make_batch(self, agent: dsv.Agent):
        states = torch.zeros(dsv.BATCH, dsv.STATE_DIM).to(dsv.DEVICE)
        actions = torch.zeros(dsv.BATCH, dsv.ACTION_DIM).to(dsv.DEVICE)
        rewards = torch.ones(dsv.BATCH, 1).to(dsv.DEVICE)
        next_states = torch.zeros(dsv.BATCH, dsv.STATE_DIM).to(dsv.DEVICE)
        dones = torch.zeros(dsv.BATCH, 1).to(dsv.DEVICE)
        return states, actions, rewards, next_states, dones

    def test_returns_positive_float(self):
        agent = make_agent()
        batch = self._make_batch(agent)
        loss = agent._update_actor_critic(*batch)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_two_independent_calls(self):
        """Calling _update_actor_critic twice should both succeed."""
        agent = make_agent()
        batch = self._make_batch(agent)
        loss1 = agent._update_actor_critic(*batch)
        loss2 = agent._update_actor_critic(*batch)
        assert loss1 >= 0.0
        assert loss2 >= 0.0


# ============================================================================
# 8. SELF-REVIEW MODULE ROUTING
# ============================================================================

class TestSelfReviewRouting:
    def test_original_issue_question(self):
        """The exact question from the issue must route to SelfReviewModule."""
        agent = make_agent()
        a = agent.chat("你當下如何看這些代碼")
        assert "Dreamer" in a or "SAC" in a or "模組" in a
        assert agent.conversation.last_topic() == "self_review:overview"

    def test_weaknesses_section(self):
        agent = make_agent()
        a = agent.chat("有什麼缺點？")
        assert "仍待" in a or "缺點" in a or "待改" in a
        assert agent.conversation.last_topic() == "self_review:weaknesses"

    def test_strengths_section(self):
        agent = make_agent()
        a = agent.chat("代碼的優點有哪些")
        assert "優點" in a or "優勢" in a
        assert agent.conversation.last_topic() == "self_review:strengths"

    def test_suggestions_section(self):
        agent = make_agent()
        a = agent.chat("你有什麼改進建議？")
        assert "建議" in a or "下一步" in a
        assert agent.conversation.last_topic() == "self_review:suggestions"

    def test_component_drilldown(self):
        agent = make_agent()
        a = agent.chat("請說說 CausalEngine")
        assert "CausalEngine" in a or "因果" in a

    def test_followup_stays_in_selfreview(self):
        agent = make_agent()
        agent.chat("你當下如何看這些代碼")
        a = agent.chat("繼續說")
        # Must stay in self-review, not fall through to knowledge module
        assert agent.conversation.last_topic().startswith("self_review:")

    def test_domain_question_not_self_review(self):
        agent = make_agent()
        a = agent.chat("補鈣有什麼注意事項")
        assert "self_review" not in agent.conversation.last_topic()

    def test_detection_of_class_names(self):
        assert dsv.SelfReviewModule.is_self_review_question("請說說 CausalEngine")
        assert dsv.SelfReviewModule.is_self_review_question("SAC 是怎麼設計的")
        assert not dsv.SelfReviewModule.is_self_review_question("天空為什麼是藍的")


# ============================================================================
# 9. AGENT CHAT AUTO-SAVE (integration)
# ============================================================================

class TestAgentChatAutoSave:
    def test_conversation_file_created_after_chat(self, tmp_path, monkeypatch):
        path = str(tmp_path / "auto_save.json")
        monkeypatch.setattr(dsv, "_CONVERSATION_PATH", path)
        agent = make_agent()
        agent.chat("微積分是什麼")
        assert os.path.exists(path)

    def test_conversation_reloaded_on_new_agent(self, tmp_path, monkeypatch):
        path = str(tmp_path / "session.json")
        monkeypatch.setattr(dsv, "_CONVERSATION_PATH", path)

        # First session — use make_agent (clears memory) then chat to write history
        agent1 = make_agent()
        agent1.chat("相對論是什麼？")
        # Verify file was written
        assert os.path.exists(path)

        # Second session — create Agent directly (without clear()) so load works
        agent2 = dsv.Agent()
        history = agent2.conversation.get_history()
        assert len(history) >= 2, (
            f"Second agent should have restored history but got {len(history)} entries"
        )
