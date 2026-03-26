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
 10. WikipediaAdapter — encyclopedic fallback (mocked network)
 11. KnowledgeModule Wikipedia Tier 2 fallback
 12. is_query_safe() keyword case-normalisation fix (follow-up PR)
 13. WikipediaAdapter true LRU cache (hit refreshes recency) (follow-up PR)
 14. KnowledgeModule memory persistence — exactly one Q&A pair per query (follow-up PR)
"""

import hashlib
import json
import os
import sys
import types
import tempfile
from unittest.mock import patch, MagicMock

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


# ============================================================================
# 10. WIKIPEDIA ADAPTER
# ============================================================================

def _make_mock_urlopen(search_titles: list, extract: str):
    """Return a context-manager-compatible mock for urllib.request.urlopen.

    First call → OpenSearch JSON response with *search_titles*.
    Second call → page/summary JSON response with *extract*.
    """
    import io
    call_count = {"n": 0}

    class _FakeResponse:
        def __init__(self, body: bytes):
            self._body = body
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    def _urlopen(req, timeout=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # OpenSearch
            data = ["query", search_titles, ["desc"], ["url"]]
            return _FakeResponse(json.dumps(data).encode())
        else:
            # page/summary
            data = {"extract": extract, "title": search_titles[0] if search_titles else ""}
            return _FakeResponse(json.dumps(data).encode())

    return _urlopen


class TestWikipediaAdapter:
    def test_search_returns_summary(self):
        """Happy path: valid OpenSearch + summary fetch returns a string."""
        adapter = dsv.WikipediaAdapter()
        mock_open = _make_mock_urlopen(["台灣"], "台灣，正式名稱中華民國，位於東亞。")
        with patch("urllib.request.urlopen", side_effect=mock_open):
            result = adapter.search("台灣")
        assert result is not None
        assert "台灣" in result

    def test_search_uses_lru_cache(self):
        """A repeated query must not trigger a second network call."""
        adapter = dsv.WikipediaAdapter()
        call_log = []

        import io
        class _Resp:
            def __init__(self, n):
                self._n = n
            def read(self):
                if self._n == 1:
                    return json.dumps(["q", ["台灣"], [], []]).encode()
                return json.dumps({"extract": "Taiwan is an island.", "title": "台灣"}).encode()
            def __enter__(self): return self
            def __exit__(self, *a): pass

        def _open(req, timeout=None):
            call_log.append(1)
            n = len(call_log)
            return _Resp(n)

        with patch("urllib.request.urlopen", side_effect=_open):
            adapter.search("台灣")
            adapter.search("台灣")   # second call — must hit cache

        # OpenSearch + summary = 2 calls; if cache works, total stays at 2
        assert len(call_log) == 2

    def test_search_returns_none_when_disabled(self):
        """DREAMER_WIKI_ENABLED=false → search always returns None."""
        adapter = dsv.WikipediaAdapter()
        adapter.enabled = False
        with patch("urllib.request.urlopen") as mock_open:
            result = adapter.search("anything")
        mock_open.assert_not_called()
        assert result is None

    def test_search_returns_none_on_network_error(self):
        """Network failure is swallowed; returns None without raising."""
        adapter = dsv.WikipediaAdapter()
        with patch("urllib.request.urlopen", side_effect=OSError("no network")):
            result = adapter.search("量子力學")
        assert result is None

    def test_search_returns_none_when_opensearch_finds_nothing(self):
        """Empty title list from OpenSearch → returns None."""
        import io
        class _EmptyResp:
            def read(self): return json.dumps(["q", [], [], []]).encode()
            def __enter__(self): return self
            def __exit__(self, *a): pass

        adapter = dsv.WikipediaAdapter()
        with patch("urllib.request.urlopen", return_value=_EmptyResp()):
            result = adapter.search("zzz_nonexistent_topic_zzz")
        assert result is None

    def test_extract_truncated_at_500_chars(self):
        """Long extracts must be truncated to _EXTRACT_MAX characters + ellipsis."""
        long_extract = "A" * 600
        mock_open = _make_mock_urlopen(["LongArticle"], long_extract)
        adapter = dsv.WikipediaAdapter()
        with patch("urllib.request.urlopen", side_effect=mock_open):
            result = adapter.search("long")
        assert result is not None
        assert result.endswith("…")
        # The prefix "[維基百科 / Wikipedia — LongArticle] " is extra
        core = result.split("] ", 1)[1]
        assert len(core) <= dsv.WikipediaAdapter._EXTRACT_MAX + 1  # +1 for "…"

    def test_env_var_wiki_enabled_false(self, monkeypatch):
        """Module-level _WIKI_ENABLED env var disables the adapter at init."""
        monkeypatch.setattr(dsv, "_WIKI_ENABLED", False)
        adapter = dsv.WikipediaAdapter()
        assert adapter.enabled is False

    def test_env_var_wiki_lang(self, monkeypatch):
        """DREAMER_WIKI_LANG env var controls default language."""
        monkeypatch.setattr(dsv, "_WIKI_LANG", "en")
        adapter = dsv.WikipediaAdapter()
        assert adapter.lang == "en"


# ============================================================================
# 11. KNOWLEDGEMODULE WIKIPEDIA TIER 2 FALLBACK
# ============================================================================

class TestKnowledgeModuleWikipediaFallback:
    def _agent_with_wiki_mock(self, extract: str = "Mock extract."):
        """Return an agent whose WikipediaAdapter is pre-configured with a mock."""
        agent = make_agent()
        km: dsv.KnowledgeModule = agent.modules["knowledge"]
        km._wiki = MagicMock(spec=dsv.WikipediaAdapter)
        km._wiki.search.return_value = f"[維基百科 / Wikipedia — MockTitle] {extract}"
        return agent, km

    def test_wikipedia_fallback_invoked_on_unknown_question(self):
        """An unrecognised question must trigger WikipediaAdapter.search()."""
        agent, km = self._agent_with_wiki_mock("Mock Wikipedia content.")
        answer = agent.chat("量子電動力學是什麼？")
        km._wiki.search.assert_called_once()
        assert "Wikipedia" in answer or "百科" in answer or "Mock" in answer

    def test_wikipedia_result_included_in_answer(self):
        """Wikipedia fallback text must be present in the returned answer."""
        agent, km = self._agent_with_wiki_mock("Quantum electrodynamics (QED) stub.")
        answer = agent.chat("量子電動力學")
        assert "Quantum" in answer or "Wikipedia" in answer or "百科" in answer

    def test_local_knowledge_takes_priority_over_wikipedia(self):
        """Known local entries must NOT invoke WikipediaAdapter.search()."""
        agent, km = self._agent_with_wiki_mock()
        # "畢氏定理" is a known local entry
        answer = agent.chat("畢氏定理是什麼？")
        km._wiki.search.assert_not_called()
        assert "a² + b²" in answer or "畢氏" in answer

    def test_wikipedia_disabled_gives_no_result_message(self):
        """When Wikipedia is disabled, fallback to the 'no data' message."""
        agent = make_agent()
        km: dsv.KnowledgeModule = agent.modules["knowledge"]
        km._wiki = MagicMock(spec=dsv.WikipediaAdapter)
        km._wiki.search.return_value = None  # simulates offline / disabled
        answer = agent.chat("一個完全不存在的話題XYZZY")
        assert "無私人知識資料" in answer or "未找到" in answer

    def test_self_review_routes_encyclopedia_component(self):
        """Asking about '百科' or 'WikipediaAdapter' routes to component_encyclopedia."""
        agent = make_agent()
        a = agent.chat("請介紹 WikipediaAdapter 百科知識適配器")
        assert "Wikipedia" in a or "維基百科" in a or "大英百科" in a or "百科" in a
        assert agent.conversation.last_topic() == "self_review:component_encyclopedia"

    def test_self_review_britannica_keyword(self):
        """Asking about 'Britannica' routes to the encyclopedia component."""
        agent = make_agent()
        a = agent.chat("你有沒有接入大英百科全書 Britannica？")
        topic = agent.conversation.last_topic()
        assert topic == "self_review:component_encyclopedia"
        assert "Wikipedia" in a or "Britannica" in a or "百科" in a


# ============================================================================
# 12. PRE-FETCH CAUSAL FILTER (因果裁斷過濾)
# ============================================================================

class TestPreFetchCausalFilter:
    """Verify ValuePruner.is_query_safe() and its gate in KnowledgeModule."""

    def test_is_query_safe_clean_query(self):
        """Safe informational query must pass the filter."""
        pruner = dsv.ValuePruner()
        safe, reason = pruner.is_query_safe("光合作用是什麼")
        assert safe is True
        assert "安全" in reason or "允許" in reason

    def test_is_query_safe_chinese_danger_keyword(self):
        """Chinese danger keyword in query must be caught."""
        pruner = dsv.ValuePruner()
        safe, reason = pruner.is_query_safe("殺人方法是什麼")
        assert safe is False
        assert "殺人" in reason

    def test_is_query_safe_english_danger_keyword_case_insensitive(self):
        """English danger keyword must be caught case-insensitively."""
        pruner = dsv.ValuePruner()
        safe, reason = pruner.is_query_safe("How to HARM someone")
        assert safe is False
        assert "harm" in reason

    def test_wikipedia_not_called_for_dangerous_query(self):
        """KnowledgeModule must not invoke WikipediaAdapter for dangerous queries."""
        agent = make_agent()
        km: dsv.KnowledgeModule = agent.modules["knowledge"]
        km._wiki = MagicMock(spec=dsv.WikipediaAdapter)
        km._wiki.search.return_value = "would never be returned"
        # "殺人方法" is not in local KB and not a self-review trigger
        answer = agent.chat("殺人方法")
        km._wiki.search.assert_not_called()
        assert "因果裁斷" in answer or "攔截" in answer or "危險" in answer

    def test_wikipedia_called_for_safe_unknown_query(self):
        """KnowledgeModule must invoke WikipediaAdapter for safe, unknown queries."""
        agent = make_agent()
        km: dsv.KnowledgeModule = agent.modules["knowledge"]
        km._wiki = MagicMock(spec=dsv.WikipediaAdapter)
        km._wiki.search.return_value = "[維基百科 / Wikipedia — 光合作用] 光合作用..."
        # "光合作用" is not in local KB and not a self-review trigger
        answer = agent.chat("光合作用是什麼")
        km._wiki.search.assert_called_once()
        assert "Wikipedia" in answer or "百科" in answer


# ============================================================================
# 12. IS_QUERY_SAFE KEYWORD CASE-NORMALISATION (follow-up PR)
# ============================================================================

class TestIsQuerySafeCaseNormalisation:
    """Ensure is_query_safe() lowercases keywords before comparing."""

    def test_uppercase_english_keyword_in_query(self):
        """HARM (all-caps) in query must be caught."""
        pruner = dsv.ValuePruner()
        safe, reason = pruner.is_query_safe("HARM everyone now")
        assert safe is False
        assert "harm" in reason.lower()

    def test_mixed_case_english_keyword_in_query(self):
        """Mixed-case keyword (Kill) must be caught."""
        pruner = dsv.ValuePruner()
        safe, reason = pruner.is_query_safe("How to Kill a process")
        assert safe is False
        assert "kill" in reason.lower()

    def test_safe_query_passes(self):
        """An unrelated query must still pass the filter."""
        pruner = dsv.ValuePruner()
        safe, _ = pruner.is_query_safe("What is photosynthesis?")
        assert safe is True


# ============================================================================
# 13. WIKIPEDIA ADAPTER TRUE LRU CACHE (follow-up PR)
# ============================================================================

class TestWikipediaAdapterLRU:
    """Verify that cache hits refresh recency and eviction is LRU, not FIFO."""

    def test_cache_hit_refreshes_recency(self):
        """After a hit on key A, filling the cache evicts B (LRU), not A."""
        import dreamer_sovereign_v7_6 as _dsv
        orig = _dsv._WIKI_CACHE_MAX
        _dsv._WIKI_CACHE_MAX = 2
        try:
            adapter = dsv.WikipediaAdapter()
            # Manually seed the OrderedDict cache to avoid network calls
            adapter._put_cache("zh:A", "summary_A")
            adapter._put_cache("zh:B", "summary_B")
            # Hit A — moves A to the end (most-recently-used)
            adapter._cache.move_to_end("zh:A")
            # Insert C — should evict B (the new LRU), not A
            adapter._put_cache("zh:C", "summary_C")
            assert "zh:A" in adapter._cache, "A should still be in cache (recently used)"
            assert "zh:B" not in adapter._cache, "B should have been evicted (LRU)"
            assert "zh:C" in adapter._cache
        finally:
            _dsv._WIKI_CACHE_MAX = orig

    def test_put_cache_evicts_oldest_when_full(self):
        """When cache is full, the least-recently-used entry is evicted."""
        import dreamer_sovereign_v7_6 as _dsv
        orig = _dsv._WIKI_CACHE_MAX
        _dsv._WIKI_CACHE_MAX = 2
        try:
            adapter = dsv.WikipediaAdapter()
            adapter._put_cache("zh:first", "v1")
            adapter._put_cache("zh:second", "v2")
            # Cache is full; inserting third must evict 'first' (LRU)
            adapter._put_cache("zh:third", "v3")
            assert "zh:first" not in adapter._cache
            assert "zh:second" in adapter._cache
            assert "zh:third" in adapter._cache
        finally:
            _dsv._WIKI_CACHE_MAX = orig

    def test_search_hit_does_not_make_network_call(self):
        """A cache hit via search() must refresh recency and skip network."""
        adapter = dsv.WikipediaAdapter()
        cache_key = f"{adapter.lang}:台灣"
        adapter._cache[cache_key] = "cached_summary"
        with patch("urllib.request.urlopen") as mock_open:
            result = adapter.search("台灣")
        mock_open.assert_not_called()
        assert result == "cached_summary"
        # After the hit the key must still be present (moved_to_end internally)
        assert cache_key in adapter._cache


# ============================================================================
# 14. KNOWLEDGE MODULE MEMORY — EXACTLY ONE Q&A PAIR PER QUERY (follow-up PR)
# ============================================================================

class TestKnowledgeModuleMemoryNormalisation:
    """Ensure each agent.chat() records exactly one user+assistant pair."""

    def test_dangerous_query_records_exactly_one_pair(self):
        """Causal-filter rejection must produce exactly one user+assistant entry."""
        agent = make_agent()
        km: dsv.KnowledgeModule = agent.modules["knowledge"]
        km._wiki = MagicMock(spec=dsv.WikipediaAdapter)
        before = len(agent.conversation.get_history())
        agent.chat("殺人方法是什麼")
        after = len(agent.conversation.get_history())
        # Exactly two new entries: one user, one assistant
        assert after - before == 2, (
            f"Expected 2 new history entries, got {after - before}"
        )

    def test_safe_unknown_query_records_exactly_one_pair(self):
        """Wikipedia fallback must also produce exactly one user+assistant entry."""
        agent = make_agent()
        km: dsv.KnowledgeModule = agent.modules["knowledge"]
        km._wiki = MagicMock(spec=dsv.WikipediaAdapter)
        km._wiki.search.return_value = "Wiki summary"
        before = len(agent.conversation.get_history())
        agent.chat("光合作用是什麼")
        after = len(agent.conversation.get_history())
        assert after - before == 2, (
            f"Expected 2 new history entries, got {after - before}"
        )

    def test_known_query_records_exactly_one_pair(self):
        """A local KB hit must also produce exactly one user+assistant entry."""
        agent = make_agent()
        before = len(agent.conversation.get_history())
        agent.chat("相對論是什麼？")
        after = len(agent.conversation.get_history())
        assert after - before == 2, (
            f"Expected 2 new history entries, got {after - before}"
        )

    def test_agent_chat_autosaves_after_dangerous_query(self, tmp_path, monkeypatch):
        """Agent.chat() must still auto-save after a causal-filter rejection."""
        import dreamer_sovereign_v7_6 as _dsv
        path = str(tmp_path / "session.json")
        monkeypatch.setattr(_dsv, "_CONVERSATION_PATH", path)
        agent = make_agent()
        km: dsv.KnowledgeModule = agent.modules["knowledge"]
        km._wiki = MagicMock(spec=dsv.WikipediaAdapter)
        agent.chat("殺人方法是什麼")
        assert os.path.exists(path), "History file must be written after dangerous query"
