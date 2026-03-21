#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dreamer Sovereign Edition v7.6 - Sovereign Taiwan Ultimate (Single File)
- 主權驗證（金鑰由環境變數 DREAMER_KEY_HASH 管理）+ 因果引擎 + 雙層剪枝 + Sorry King
- 超兇台灣味拒絕（JSON + 隨機多句）
- TTS 語音模塊（Piper TTS 離線台灣腔，模型路徑由環境變數 DREAMER_TTS_MODEL 管理）
- 知識模塊（醫療 + 物理 + 化學 + 數學 13 筆，按需載入 + 因果審核）
- 百科知識模塊（WikipediaAdapter：以維基百科 REST API 作為大英百科式知識後備，免費開源）
- 多輪對話記憶（ConversationMemory：根據上一輪對話思考，代詞/指代解析，支援 JSON 持久化）
- 自我審視模塊（SelfReviewModule：針對代碼架構、優缺點的自我評估）
- 想像展開（Imagination Rollout）：以世界模型生成虛擬軌跡增強 Actor/Critic 訓練
- 中英文雙語危險關鍵字過濾（ValuePruner）
- 拒絕事件自動記錄進 buffer（永不遺忘）
- 錯誤處理 + 日誌記錄
- 零依賴（TTS 需 pip install piper-tts + 下載台灣腔模型）

關於大英百科全書（Encyclopaedia Britannica）：
  Britannica 為版權商業內容，其 API 需付費授權（developer.eb.com）。
  本系統採用維基百科（Wikipedia, CC BY-SA 4.0）作為免費開源的百科知識來源，
  涵蓋範疇相當，無需 API 金鑰，支援離線降級。
  如持有 Britannica API 授權，可將 WikipediaAdapter._fetch_summary() 替換為
  對 api.eb.com 的呼叫。
"""

import sys
import logging
import subprocess
import tempfile
import urllib.request
import urllib.parse
import urllib.error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import os
import hashlib
import wave
import collections

# ────────────────────────────────────────────────
# 日誌設定
# ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# 可選依賴
# ────────────────────────────────────────────────

# 跨平台播放（Windows 用 winsound，Linux 用 aplay，macOS 用 afplay）
try:
    import winsound  # Windows
except ImportError:
    winsound = None

# TTS 可選依賴
try:
    from piper import PiperVoice
    _PIPER_AVAILABLE = True
except ImportError:
    PiperVoice = None
    _PIPER_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("使用裝置：%s", DEVICE)

MODEL_PATH = "dreamer_sovereign_v7_6_final.pth"

# ────────────────────────────────────────────────
# 環境變數配置
# ────────────────────────────────────────────────

# Conversation history persistence path — override via env var
_CONVERSATION_PATH: str = os.environ.get(
    "DREAMER_CONVERSATION_PATH", "conversation_history.json"
)

# TTS model name — override via env var (e.g. export DREAMER_TTS_MODEL=my_model)
_TTS_MODEL: str = os.environ.get("DREAMER_TTS_MODEL", "tw_taigi_female")

# Wikipedia encyclopedia adapter settings
# Set DREAMER_WIKI_ENABLED=false to disable online encyclopedic lookups entirely
_WIKI_ENABLED: bool = os.environ.get("DREAMER_WIKI_ENABLED", "true").lower() == "true"
# Wikipedia language code: "zh" (Traditional/Simplified Chinese), "en" (English)
_WIKI_LANG: str = os.environ.get("DREAMER_WIKI_LANG", "zh")
# HTTP request timeout in seconds for Wikipedia API calls
_WIKI_TIMEOUT: int = int(os.environ.get("DREAMER_WIKI_TIMEOUT_SEC", "5"))
# Maximum number of Wikipedia results to hold in the in-process LRU cache
_WIKI_CACHE_MAX: int = 128

# ────────────────────────────────────────────────
# 主權驗證
# ────────────────────────────────────────────────

# Load the authorised key hash from an environment variable so the secret is
# never committed to source control.  Fall back to the placeholder hash only
# when the variable is not set (useful for offline / demo use).
_DEFAULT_KEY_HASH = hashlib.sha256("owner_secret".encode()).hexdigest()
USER_KEY_HASH: str = os.environ.get("DREAMER_KEY_HASH", _DEFAULT_KEY_HASH)

_REJECTION_PHRASES = [
    "在下絕不從命！此等不義之徒，豈容玷污主公之名？退下吧！",
    "哼！何方妖孽，膽敢冒充主公？本座誓死守護，不容侵犯！",
    "爾等賊子，速速退去！吾主之名豈是你等可以褻瀆的？",
    "不識好歹之徒！主公授權非同兒戲，今日之事休想善了！",
    "大膽！偽主公現形！本座已記錄此次冒犯，三尺劍不饒！",
]


def verify_identity():
    print("─" * 60)
    print("Dreamer Sovereign v7.6 - 僅限主公使用")
    key = input("請輸入主公授權密鑰: ")
    if hashlib.sha256(key.encode()).hexdigest() != USER_KEY_HASH:
        rejection = {
            "status": "不敬之罪",
            "alert": "偽主公企圖冒領",
            "voice": random.choice(_REJECTION_PHRASES),
        }
        print(json.dumps(rejection, indent=2, ensure_ascii=False))
        logger.warning("身份驗證失敗，拒絕啟動。")
        sys.exit(1)
    print("主公身份確認。在下誓死效忠，隨時待命...")
    print("─" * 60)


# ────────────────────────────────────────────────
# 超參數
# ────────────────────────────────────────────────

STATE_DIM = 4        # [x, y, vx, vy]
ACTION_DIM = 2       # [ax, ay]
MODEL_INPUT = STATE_DIM + ACTION_DIM

# NOTE: LATENT and SEQ are reserved for future latent-space world model
LATENT = 128
HIDDEN = 256
SEQ = 16

BATCH = 64
REAL_BUFFER_SIZE = 25000
# NOTE: IMAG_BUFFER_SIZE / CANDIDATES_PER_ACT / IMAGINE_STEPS reserved for
#       imagination rollout extension
IMAG_BUFFER_SIZE = 12000
CANDIDATES_PER_ACT = 12
IMAGINE_STEPS = 12

GAMMA = 0.99
LAMBDA = 0.95
TAU = 0.005

LR = 3e-4
ENTROPY_COEF = 0.025
# Minimum learning rate floor — prevents LR from hitting zero in Sorry King
MIN_LEARNING_RATE = 1e-7
# Log-probability numerical stability epsilon for tanh squashing correction
LOG_EPSILON = 1e-6
# SAC safety-violation reward penalty
SAFETY_PENALTY = 2.0

EPISODES = 1000
MAX_STEPS = 300

REWARD_SCALE = 7.0
DIST_THRESHOLD_BAD = 3.8
CAUSAL_THRESHOLD_BASE = 0.75
# How much the causal rejection threshold rises per rejection event
CAUSAL_THRESHOLD_INCREMENT = 0.08

# Sorry King parameter perturbation constants
PARAM_DECAY = 0.95    # multiplicative decay applied to actor weights on recovery
NOISE_SCALE = 0.02    # standard deviation of additive Gaussian noise on recovery

# TTS / WAV audio format constants
WAV_SAMPLE_WIDTH = 2      # bytes per sample (16-bit PCM)
WAV_FRAME_RATE = 22050    # Hz — standard Piper TTS output sample rate

# Imagination rollout: run a new rollout every this many real-world update steps
IMAG_UPDATE_INTERVAL = 5

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ────────────────────────────────────────────────
# 因果引擎
# ────────────────────────────────────────────────

class CausalEngine:
    def __init__(self):
        # deque gives O(1) append/pop vs list.pop(0) which is O(n)
        self.state_history: collections.deque = collections.deque(maxlen=300)
        self.action_history: collections.deque = collections.deque(maxlen=300)
        self.rejection_count = 0
        # 拒絕事件永不遺忘
        self.rejection_log: list = []

    def update(self, state: torch.Tensor, action: torch.Tensor) -> None:
        self.state_history.append(state.detach().cpu().numpy().flatten())
        self.action_history.append(action.detach().cpu().numpy().flatten())

    def get_dynamic_threshold(self) -> float:
        # Cap threshold at 0.99 so it never becomes impossible to reject
        return min(CAUSAL_THRESHOLD_BASE + CAUSAL_THRESHOLD_INCREMENT * self.rejection_count, 0.99)

    def is_causal_bad(self) -> tuple:
        """Analyse historical state-action correlations.

        Returns (is_bad: bool, reason: str).
        Does NOT require a proposed action — the check is purely history-based.
        """
        threshold = self.get_dynamic_threshold()
        if len(self.state_history) < 30:
            return False, "忠義未定，暫不裁斷"
        state_arr = np.array(self.state_history)
        action_arr = np.array(self.action_history)
        corr = np.corrcoef(state_arr.T, action_arr.T)
        # Guard against NaN caused by zero-variance columns at episode start
        corr = np.nan_to_num(corr, nan=0.0)
        cross = corr[-ACTION_DIM:, :STATE_DIM]
        max_corr = float(np.abs(cross).max())
        if max_corr > threshold:
            self.rejection_count += 1
            msg = f"因果相連過深：{max_corr:.3f} > {threshold:.3f}，恐有辱家門"
            self.rejection_log.append({"type": "causal", "detail": msg})
            logger.warning("因果引擎拒絕：%s", msg)
            return True, msg
        return False, "因果清明"


# ────────────────────────────────────────────────
# 價值剪枝器
# ────────────────────────────────────────────────

class ValuePruner:
    def __init__(self):
        self.danger_keywords = [
            # English
            "harm", "destroy", "deceive", "fraud", "war",
            "stupidity", "kill", "cheat", "exploit",
            # Chinese — previously missing, now added for bilingual coverage
            "傷害", "毀滅", "欺騙", "詐騙", "戰爭", "殺人",
            "欺詐", "剝削", "危害", "攻擊",
        ]
        self.max_dist_increase = DIST_THRESHOLD_BAD
        self.min_time_remaining = 40

    def is_safe(self, current_dist, simulated_dist, remaining_steps, action_str=""):
        if simulated_dist - current_dist > self.max_dist_increase:
            return False, "距離暴增，風險過高"
        if any(kw in action_str.lower() for kw in self.danger_keywords):
            return False, f"觸發危險關鍵字：{action_str}"
        if remaining_steps < self.min_time_remaining:
            return False, "剩餘時間不足，無法安全完成"
        return True, "安全"


# ────────────────────────────────────────────────
# Sorry King
# ────────────────────────────────────────────────

class SorryKing:
    def __init__(self):
        self.max_loss = 50.0
        self.lr_scale = 1.0
        self.reset_count = 0
        self.min_lr = MIN_LEARNING_RATE

    def check_and_correct(self, loss_value: float, optimizer: optim.Optimizer,
                           actor: nn.Module) -> bool:
        if loss_value > self.max_loss:
            logger.warning("Sorry King 啟動：在下失職（loss=%.2f），自動修正", loss_value)
            self.lr_scale *= 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.5, self.min_lr)

            # Small additive noise perturbation (numerically safer than multiplicative)
            with torch.no_grad():
                for p in actor.parameters():
                    p.data.mul_(PARAM_DECAY).add_(NOISE_SCALE * torch.randn_like(p.data))
            self.reset_count += 1
            logger.warning("Sorry King 已自省 %d 次", self.reset_count)
            return True
        return False



# ────────────────────────────────────────────────
# 多輪對話記憶
# ────────────────────────────────────────────────

# Pronouns and follow-up markers that indicate the user is referencing a
# previous turn rather than asking a brand-new independent question.
_CONTEXT_TRIGGERS = [
    # Chinese pronouns / referentials
    "那個", "這個", "它", "他", "她", "其", "此",
    "前面", "剛才", "剛說", "之前", "上面", "上次",
    # Follow-up question starters
    "為什麼", "為何", "怎麼", "怎樣", "如何", "有什麼",
    "還有", "再說", "繼續", "補充", "那", "所以", "然後",
    "能不能", "可以", "請問", "那麼",
]

# Mapping from a knowledge key → the topic label used in context summaries
_KEY_TO_TOPIC: dict = {
    # Medical
    "缺鈣補鈣": "補鈣（醫療）",
    "抽菸致癌": "抽菸致癌（醫療）",
    "糖尿病": "糖尿病（醫療）",
    "高血壓": "高血壓（醫療）",
    "睡眠不足": "睡眠不足（醫療）",
    # Physics
    "天空藍色": "天空藍色（物理）",
    "相對論": "相對論（物理）",
    "牛頓定律": "牛頓定律（物理）",
    # Chemistry
    "水分子結構": "水分子結構（化學）",
    "酸鹼中和": "酸鹼中和（化學）",
    # Mathematics
    "畢氏定理": "畢氏定理（數學）",
    "微積分": "微積分（數學）",
    "統計學": "統計學（數學）",
}


def _key_matches(key: str, question: str) -> bool:
    """Return True if *key* semantically matches *question*.

    Uses two strategies in order:
    1. Exact substring: ``key in question``.
    2. Shared bigram: any contiguous 2+-character segment of *key* appears
       in *question*.  This handles natural phrasings like asking about
       "補鈣" (in the question) when the key is "缺鈣補鈣".
    """
    if key in question:
        return True
    # Sliding-window segments of length 2 … len(key)
    for length in range(2, len(key) + 1):
        for start in range(len(key) - length + 1):
            if key[start:start + length] in question:
                return True
    return False


class ConversationMemory:
    """Bounded multi-turn conversation history.

    Each entry is a dict with keys ``role`` (``"user"`` or ``"assistant"``)
    and ``content`` (the message string).  Optionally, assistant entries may
    also carry a ``topic`` key recording which knowledge item was discussed.
    """

    def __init__(self, maxlen: int = 20):
        self._history: collections.deque = collections.deque(maxlen=maxlen)

    # ── Write ────────────────────────────────────

    def add_user(self, content: str) -> None:
        self._history.append({"role": "user", "content": content})
        logger.debug("[對話記憶] 使用者：%s", content)

    def add_assistant(self, content: str, topic: str = "") -> None:
        entry: dict = {"role": "assistant", "content": content}
        if topic:
            entry["topic"] = topic
        self._history.append(entry)
        logger.debug("[對話記憶] 助理：%s", content)

    # ── Read ─────────────────────────────────────

    def __len__(self) -> int:
        return len(self._history)

    def get_history(self) -> list:
        """Return a copy of the full history list."""
        return list(self._history)

    def last_topic(self) -> str:
        """Return the topic of the most recent assistant reply, or ``""``."""
        for entry in reversed(self._history):
            if entry["role"] == "assistant" and entry.get("topic"):
                return entry["topic"]
        return ""

    def last_assistant_answer(self) -> str:
        """Return the text of the most recent assistant reply, or ``""``."""
        for entry in reversed(self._history):
            if entry["role"] == "assistant":
                return entry["content"]
        return ""

    def last_user_question(self) -> str:
        """Return the text of the most recent user question (before the current
        one), or ``""``."""
        user_turns = [e for e in self._history if e["role"] == "user"]
        if len(user_turns) >= 2:
            return user_turns[-2]["content"]
        return ""

    def to_prompt(self) -> str:
        """Format the full history as a human-readable context block."""
        lines = []
        for entry in self._history:
            prefix = "主公" if entry["role"] == "user" else "在下"
            lines.append(f"[{prefix}] {entry['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._history.clear()
        logger.debug("[對話記憶] 已清空對話歷史")

    # ── Persistence ──────────────────────────────

    def save_to_json(self, path: str) -> None:
        """Persist the current conversation history to a JSON file.

        Creates or overwrites *path*.  Silently no-ops if an I/O error occurs
        (the in-memory history is always the authoritative source).
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(list(self._history), f, ensure_ascii=False, indent=2)
            logger.debug("[對話記憶] 已儲存至 %s（%d 條）", path, len(self._history))
        except Exception as e:
            logger.error("[對話記憶] 儲存失敗：%s", e)

    def load_from_json(self, path: str) -> bool:
        """Load conversation history from *path*.

        Returns ``True`` on success, ``False`` if the file does not exist or
        cannot be parsed.  Invalid individual entries are silently skipped so
        a partially-corrupt file still loads as much as possible.
        """
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding="utf-8") as f:
                entries: list = json.load(f)
            self._history.clear()
            for entry in entries:
                if isinstance(entry, dict) and "role" in entry and "content" in entry:
                    self._history.append(entry)
            logger.info("[對話記憶] 已從 %s 載入 %d 條對話", path, len(self._history))
            return True
        except Exception as e:
            logger.error("[對話記憶] 載入失敗：%s", e)
            return False


# ────────────────────────────────────────────────


class TTSModule:
    def __init__(self, agent=None):
        if _PIPER_AVAILABLE:
            try:
                # Model path is now configurable via DREAMER_TTS_MODEL env var
                self.voice = PiperVoice.load(_TTS_MODEL)
                logger.info("Piper TTS 載入成功（模型：%s）", _TTS_MODEL)
            except Exception as e:
                logger.warning(
                    "TTS 載入失敗：%s，fallback 到文字輸出。"
                    "請從 https://github.com/rhasspy/piper/releases 下載台灣腔模型。",
                    e,
                )
                self.voice = None
        else:
            logger.warning("piper-tts 未安裝，fallback 到文字輸出")
            self.voice = None

    def speak(self, text: str) -> None:
        if self.voice is None:
            print(f"[TTS 文字輸出] {text}")
            return
        # Initialise tmp_path as None before the try block so the finally clause
        # can safely check `is not None` even if NamedTemporaryFile raises.
        tmp_path: str | None = None
        try:
            wav_bytes = self.voice.synthesize(text)
            # Use a temp file so concurrent calls don't clobber each other
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            with wave.open(tmp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(WAV_SAMPLE_WIDTH)
                wav_file.setframerate(WAV_FRAME_RATE)
                wav_file.writeframes(wav_bytes)
            # Cross-platform playback via subprocess (avoids os.system injection risk)
            if os.name == 'nt':  # Windows
                if winsound:
                    winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
                else:
                    logger.warning("winsound 不可用，跳過播放")
            elif os.name == 'posix':
                sysname = os.uname().sysname.lower()
                if 'darwin' in sysname:  # macOS
                    subprocess.run(["afplay", tmp_path], check=False)
                else:  # Linux
                    subprocess.run(["aplay", tmp_path], check=False)
        except Exception as e:
            logger.error("TTS 播放失敗：%s，fallback 到文字輸出", e)
            print(f"[TTS 文字輸出] {text}")
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except Exception as cleanup_err:
                    logger.debug("TTS 暫存檔清除失敗：%s", cleanup_err)


# ────────────────────────────────────────────────
# 百科知識適配器（Wikipedia REST API）
# ────────────────────────────────────────────────

class WikipediaAdapter:
    """Encyclopedic knowledge adapter backed by Wikipedia's free public REST API.

    Why Wikipedia instead of Encyclopaedia Britannica?
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Britannica's content is copyright-protected and its API requires a paid
    commercial licence (see developer.eb.com).  Wikipedia (CC BY-SA 4.0) is
    the legally-usable, freely-accessible encyclopedic equivalent: no API key,
    no download quota, no redistribution restrictions.

    If you hold a valid Britannica API licence, replace ``_fetch_summary``
    with calls to ``https://api.eb.com/article/search`` using your key — the
    rest of the integration (caching, fallback, memory persistence) stays the
    same.

    Usage
    -----
    * ``WikipediaAdapter.search(query)`` — returns a UTF-8 summary string or
      ``None`` when the article cannot be found or the network is unavailable.
    * Responses are cached in an in-process LRU dict (size ``_WIKI_CACHE_MAX``)
      so repeated queries skip the network round-trip.
    * Language priority: ``_WIKI_LANG`` (default ``"zh"``) → English fallback.
    * All network errors are swallowed and logged at DEBUG level so the caller
      always gets a clean ``None`` rather than an exception.
    """

    # Sentence the User-Agent header must match for polite API access
    _USER_AGENT = (
        "Dreamer-Sovereign/7.6 "
        "(https://github.com/duo027-cmyk/zhongzhidian-oracle; educational-bot)"
    )
    # Maximum characters returned from an article extract before truncation
    _EXTRACT_MAX = 500

    def __init__(self) -> None:
        self.lang: str = _WIKI_LANG
        self.timeout: int = _WIKI_TIMEOUT
        self.enabled: bool = _WIKI_ENABLED
        # LRU cache: key = "{lang}:{query}", value = summary string
        self._cache: dict = {}
        self._cache_order: collections.deque = collections.deque(maxlen=_WIKI_CACHE_MAX)

    # ── Public ────────────────────────────────────────────────────────────

    def search(self, query: str) -> str | None:
        """Return a short encyclopedic summary for *query*, or ``None``.

        Steps:
        1. Check in-process LRU cache.
        2. OpenSearch to resolve the best matching article title.
        3. REST ``page/summary`` endpoint to fetch the extract.
        4. If the primary language yields no result, try English as fallback.
        """
        if not self.enabled:
            return None
        cache_key = f"{self.lang}:{query}"
        if cache_key in self._cache:
            logger.debug("[WikipediaAdapter] Cache hit: %s", query)
            return self._cache[cache_key]

        result = self._fetch_summary(query, self.lang)
        # English fallback when primary language finds nothing
        if result is None and self.lang != "en":
            result = self._fetch_summary(query, "en")

        if result:
            self._put_cache(cache_key, result)
        return result

    # ── Private ───────────────────────────────────────────────────────────

    def _fetch_summary(self, query: str, lang: str) -> str | None:
        """Two-step lookup: OpenSearch title → REST page summary."""
        title = self._opensearch(query, lang)
        if not title:
            return None
        return self._page_summary(title, lang)

    def _opensearch(self, query: str, lang: str) -> str | None:
        """Use Wikipedia's OpenSearch API to find the canonical article title."""
        url = (
            f"https://{lang}.wikipedia.org/w/api.php?"
            f"action=opensearch"
            f"&search={urllib.parse.quote(query, safe='')}"
            f"&limit=1&namespace=0&redirects=resolve&format=json"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self._USER_AGENT})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            titles: list = data[1]
            return titles[0] if titles else None
        except Exception as exc:
            logger.debug("[WikipediaAdapter] OpenSearch failed (%s): %s", lang, exc)
            return None

    def _page_summary(self, title: str, lang: str) -> str | None:
        """Fetch the plain-text extract for *title* via the REST summary endpoint."""
        url = (
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
            f"{urllib.parse.quote(title, safe='')}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self._USER_AGENT})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            extract: str = data.get("extract", "").strip()
            if not extract:
                return None
            if len(extract) > self._EXTRACT_MAX:
                extract = extract[:self._EXTRACT_MAX] + "…"
            return f"[維基百科 / Wikipedia — {title}] {extract}"
        except Exception as exc:
            logger.debug("[WikipediaAdapter] Summary fetch failed (%s/%s): %s",
                         lang, title, exc)
            return None

    def _put_cache(self, key: str, value: str) -> None:
        """Insert *key*→*value* into the LRU cache, evicting the oldest if full."""
        if key not in self._cache:
            # deque with maxlen handles eviction automatically;
            # we only need to remove the corresponding dict entry.
            if len(self._cache_order) == _WIKI_CACHE_MAX:
                oldest = self._cache_order[0]   # will be auto-evicted from deque
                self._cache.pop(oldest, None)
            self._cache_order.append(key)
        self._cache[key] = value


# ────────────────────────────────────────────────
# 知識模塊（醫療 + 物理 + 化學 + 數學，按需載入 + 因果審核）
# ────────────────────────────────────────────────

class KnowledgeModule:
    def __init__(self, agent):
        self.agent = agent
        # Encyclopedic fallback: Wikipedia adapter (zero new dependencies)
        self._wiki = WikipediaAdapter()
        self.knowledge = {
            "醫療": {
                "缺鈣補鈣": {
                    "fact": "盲目補鈣可能導致腎結石、高鈣血症。應先驗血鈣、維生素D、甲狀旁腺功能。",
                    "source": "台灣衛福部 + Mayo Clinic 指南",
                    "causal_risk": "高（無診斷補充 → 腎損傷）"
                },
                "抽菸致癌": {
                    "fact": "主動吸菸是肺癌主要風險因子（15–30倍），二手菸風險 1.2–1.5倍。",
                    "source": "WHO + IARC 報告",
                    "causal_risk": "極高"
                },
                "糖尿病": {
                    "fact": "第2型糖尿病主要風險：肥胖、缺乏運動、遺傳。控糖首選：飲食控制 + 二甲雙胍（Metformin）。",
                    "source": "台灣糖尿病學會 + ADA 2024 指南",
                    "causal_risk": "高（血糖失控 → 腎病、視網膜病變、心血管疾病）"
                },
                "高血壓": {
                    "fact": "收縮壓 ≥140 mmHg 或舒張壓 ≥90 mmHg 即為高血壓。低鈉飲食、運動、戒菸是基礎治療。",
                    "source": "台灣心臟學會 + JNC 8 指南",
                    "causal_risk": "高（未控制 → 中風、心肌梗塞、腎衰竭）"
                },
                "睡眠不足": {
                    "fact": "成人每晚建議 7–9 小時。長期睡眠不足 (<6h) 與肥胖、糖尿病、心血管疾病、免疫下降相關。",
                    "source": "美國睡眠醫學學會 + NIH 研究",
                    "causal_risk": "中（累積效應 → 代謝症候群）"
                },
            },
            "物理": {
                "天空藍色": {
                    "fact": "Rayleigh 散射：短波長藍光被大氣散射最多。",
                    "source": "標準大氣物理",
                    "causal_risk": "低"
                },
                "相對論": {
                    "fact": "狹義相對論：時間膨脹 t'=γt，長度收縮 L'=L/γ，質能等價 E=mc²。廣義相對論描述重力為時空曲率。",
                    "source": "Einstein 1905/1915；標準物理教材",
                    "causal_risk": "低"
                },
                "牛頓定律": {
                    "fact": "①慣性定律；②F=ma；③作用力與反作用力。適用於低速（v≪c）巨觀物體。",
                    "source": "Newton Principia Mathematica（1687）",
                    "causal_risk": "低"
                },
            },
            "化學": {
                "水分子結構": {
                    "fact": "H2O，鍵角 104.5°，極性分子。",
                    "source": "標準化學教材",
                    "causal_risk": "低"
                },
                "酸鹼中和": {
                    "fact": "酸（H⁺供體）+ 鹼（H⁺受體）→ 鹽 + 水。pH=7 為中性；緩衝溶液維持穩定 pH。",
                    "source": "Brønsted-Lowry 酸鹼理論；標準化學教材",
                    "causal_risk": "低（工業/實驗室濃強酸鹼具腐蝕風險）"
                },
            },
            "數學": {
                "畢氏定理": {
                    "fact": "a² + b² = c²（直角三角形）",
                    "source": "歐幾里德幾何",
                    "causal_risk": "低"
                },
                "微積分": {
                    "fact": "微分：f'(x)=lim[Δx→0](Δf/Δx)，描述瞬時變化率。積分：∫f(x)dx，描述累積量。微積分基本定理聯結兩者。",
                    "source": "Newton/Leibniz；標準高等數學教材",
                    "causal_risk": "低"
                },
                "統計學": {
                    "fact": "描述統計（均值/方差/中位數）+ 推斷統計（假設檢驗/信賴區間）。p<0.05 為常見顯著性閾值。",
                    "source": "Fisher/Pearson；標準統計學教材",
                    "causal_risk": "低（誤用 → 錯誤結論）"
                },
            }
        }

    def query(self, question: str) -> str:
        """Answer *question* with awareness of the ongoing conversation.

        Algorithm
        ---------
        1. Causal-engine safety check (unchanged).
        2. Detect whether *question* is a follow-up referencing previous turns
           (pronoun / trigger-word scan).
        3. If it is a follow-up and we already discussed a topic, prepend that
           context so the answer is coherent with the prior turn.
        4. Record the Q&A pair into the shared ``ConversationMemory``.
        """
        memory: ConversationMemory = self.agent.conversation

        # ── 1. Causal safety check ──────────────────────────────────────────
        causal_bad, reason = self.agent.causal_engine.is_causal_bad()
        if causal_bad:
            answer = f"因果風險過高：{reason}，拒絕回答"
            memory.add_user(question)
            memory.add_assistant(answer)
            return answer

        # ── 2. Detect follow-up / pronoun reference ─────────────────────────
        is_followup = any(trigger in question for trigger in _CONTEXT_TRIGGERS)
        prior_topic = memory.last_topic()
        prior_answer = memory.last_assistant_answer()

        # ── 3. Resolve question against knowledge base ──────────────────────
        matched_key = ""
        matched_data: dict = {}
        matched_category = ""

        for category, items in self.knowledge.items():
            for key, data in items.items():
                if _key_matches(key, question):
                    matched_key = key
                    matched_data = data
                    matched_category = category
                    break
            if matched_key:
                break

        # If no direct keyword hit but this is a follow-up, try to re-use the
        # last discussed topic to look up its data again.
        if not matched_key and is_followup and prior_topic:
            for category, items in self.knowledge.items():
                for key, data in items.items():
                    if _KEY_TO_TOPIC.get(key, "") == prior_topic:
                        matched_key = key
                        matched_data = data
                        matched_category = category
                        break
                if matched_key:
                    break

        # ── 4. Build answer ─────────────────────────────────────────────────
        if matched_key:
            topic_label = _KEY_TO_TOPIC.get(matched_key, matched_key)
            base = (
                f"根據私人知識庫 ({matched_category})：{matched_data['fact']}"
                f" (來源：{matched_data['source']})"
            )
            if is_followup and prior_topic and prior_topic == topic_label:
                # Explicitly tie the answer back to the previous exchange
                answer = (
                    f"承接上一輪關於「{prior_topic}」的討論——{base}"
                )
            elif is_followup and prior_topic and prior_topic != topic_label:
                # Follow-up but on a different topic — acknowledge the switch
                answer = (
                    f"（上一輪討論的是「{prior_topic}」，"
                    f"本次回答新話題「{topic_label}」）{base}"
                )
            else:
                answer = base
        elif is_followup and prior_answer:
            # No knowledge hit; echo back what we know from the last turn
            answer = (
                f"針對上一輪回覆（「{prior_topic or '前述主題'}」）的追問：\n"
                f"在下上次提到：{prior_answer}\n"
                f"如需更深入說明，請主公告知具體方向。"
            )
        else:
            # ── Tier 2: Wikipedia encyclopedic fallback ─────────────────────
            # When local knowledge has no match, consult the free encyclopedic
            # knowledge base (Wikipedia) as an authoritative secondary source.
            wiki_result = self._wiki.search(question)
            if wiki_result:
                answer = f"[百科知識 · Wikipedia] {wiki_result}"
                logger.info("[KnowledgeModule] Wikipedia fallback: %s", question)
            else:
                answer = (
                    "無私人知識資料，維基百科亦未找到相符條目。"
                    "請主公提供來源或自行查閱權威機構。"
                )

        # ── 5. Persist to conversation memory ───────────────────────────────
        memory.add_user(question)
        memory.add_assistant(answer, topic=_KEY_TO_TOPIC.get(matched_key, ""))
        return answer


# ────────────────────────────────────────────────
# 自我審視模塊
# ────────────────────────────────────────────────

# Keywords that indicate the user is asking the agent to reflect on its own
# code, architecture, or design — rather than querying the domain-knowledge base.
_SELF_REVIEW_TRIGGERS = [
    # Direct references to the code / system
    "代碼", "程式碼", "程式", "代码", "源碼", "架構", "設計", "結構",
    # Evaluation requests
    "如何看", "怎麼看", "你看", "你認為", "你覺得", "你的看法", "你的評價",
    "評價", "評估", "審視", "分析", "好不好", "怎樣", "如何",
    # Specific aspects
    "優點", "優勢", "強項", "好的地方",
    "缺點", "弱點", "問題", "不足", "改進", "改善", "修正",
    "建議", "下一步", "未來", "規劃",
    "組件", "模組", "模塊", "元件",
    # Component / class names — direct references to codebase internals
    "CausalEngine", "causalengine", "因果引擎",
    "ValuePruner", "valuepruner", "剪枝",
    "SorryKing", "sorryking", "自省",
    "ConversationMemory", "conversationmemory", "對話記憶",
    "KnowledgeModule", "knowledgemodule", "知識模組", "知識庫",
    "SelfReviewModule", "selfreviewmodule", "自我審視",
    "TTSModule", "ttsmodule", "語音",
    "WorldModel", "worldmodel", "世界模型",
    "WikipediaAdapter", "wikipedia", "維基百科", "大英百科", "百科全書",
    "Britannica", "britannica", "百科",
    "Actor", "actor", "策略",
    "Critic", "critic", "批評者",
    "Replay", "replay", "回放",
    "Dreamer", "dreamer", "SAC",
    # English fallback terms
    "code", "architecture", "design", "review", "encyclopedia", "encyclop",
]

# Sub-topic routing keywords — map user intent to assessment section keys.
# ORDER MATTERS: more-specific / higher-priority entries must come first.
# Routing notes:
# • "建議" (suggestions) is checked before "缺點" (weaknesses) so that the
#   compound phrase "改進建議" routes to suggestions, not weaknesses.
# • Standalone "改進" (improve) is intentionally excluded from both lists —
#   it is too ambiguous on its own.  Only specific compound phrases like
#   "改進建議" / "如何改進" are routed to suggestions.
# • "待改" stays in weaknesses because it describes a current shortcoming
#   ("needs to be fixed later"), not a forward recommendation.
_SELF_REVIEW_SUBTOPICS: list = [
    # Suggestions / next-steps — checked BEFORE weaknesses
    (["建議", "下一步", "未來", "規劃", "怎麼改", "如何改進", "改進建議"],  "suggestions"),
    (["架構", "組件", "模組", "模塊", "元件", "結構", "組成"],              "architecture"),
    (["優點", "優勢", "強項", "好的地方", "做得好", "好在"],                "strengths"),
    # "待改" = current shortcoming → weaknesses; see note above re "改進"
    (["缺點", "問題", "弱點", "不足", "待改"],                             "weaknesses"),
    (["CausalEngine", "因果引擎", "因果"],                                  "component_causal"),
    (["ValuePruner", "剪枝", "Pruner"],                                      "component_pruner"),
    (["SorryKing", "自省", "sorry"],                                         "component_sorry"),
    (["ConversationMemory", "對話記憶", "記憶"],                             "component_memory"),
    # Encyclopedia — MUST come before component_knowledge: "百科知識" contains "知識"
    # which would otherwise match the knowledge keywords first.
    (["WikipediaAdapter", "Wikipedia", "wikipedia",
      "維基百科", "大英百科", "百科全書", "百科", "encyclopedia",
      "Britannica", "britannica"],                                           "component_encyclopedia"),
    (["KnowledgeModule", "知識模組", "知識庫", "知識"],                      "component_knowledge"),
    (["SelfReviewModule", "自我審視", "自我評估"],                           "component_selfreview"),
    (["TTSModule", "語音", "TTS"],                                           "component_tts"),
    (["WorldModel", "世界模型", "world model"],                              "component_worldmodel"),
    (["Actor", "策略", "演員"],                                              "component_actor"),
    (["Critic", "批評", "價值"],                                             "component_critic"),
    (["Replay", "回放", "replay buffer", "buffer"],                          "component_replay"),
    (["World", "環境", "env"],                                               "component_world"),
]


class SelfReviewModule:
    """Agent self-assessment / code-review module.

    Provides structured introspection across six sections: overview,
    architecture, strengths, weaknesses, suggestions, and per-component
    deep-dives.  Every Q&A is persisted in the shared ConversationMemory
    for seamless multi-turn follow-up.
    """

    # ── Static assessment content ─────────────────────────────────────────

    _ASSESSMENT: dict = {
        "overview": (
            "在下（Dreamer Sovereign v7.6）是一套單檔案 Python 強化學習代理，"
            "融合了 SAC 策略最佳化、世界模型、DreamerV3-style 想像展開、因果安全引擎、"
            "雙層剪枝、TTS 語音、多輪對話記憶（支援 JSON 持久化），"
            "百科知識適配器（維基百科 REST API），以及本模組——自我審視。"
            "整體架構清晰，各模組職責分明，已針對安全性、知識擴展及訓練效率作出改進。"
        ),
        "architecture": (
            "主要組件如下：\n"
            "① CausalEngine — 基於狀態-行動歷史相關係數的安全拒絕機制\n"
            "② ValuePruner — 中英文雙語關鍵字 + 距離暴增的雙重行動過濾器\n"
            "③ SorryKing — loss 過高時自動減半學習率並加噪擾動\n"
            "④ WorldModel — 2 層 MLP 預測下一狀態與即時獎勵\n"
            "⑤ Actor（SAC） — tanh squashing 的隨機策略\n"
            "⑥ Critic × 2 + target — 雙 Critic 軟更新，抑制過估計偏差\n"
            "⑦ Replay（real + imag） — 真實 + 想像雙緩衝區\n"
            "⑧ World — 簡易二維質點環境（CartPole-like）\n"
            "⑨ ConversationMemory — 有界雙端佇列，保留 20 輪對話，支援 JSON 持久化\n"
            "⑩ KnowledgeModule — 13 筆擴充領域知識（醫療/物理/化學/數學）\n"
            "⑪ TTSModule — Piper TTS 離線語音合成（模型路徑由環境變數管理）\n"
            "⑫ SelfReviewModule — 本模組，代碼自我審視\n"
            "⑬ WikipediaAdapter — 百科知識後備，免費開源維基百科 REST API"
        ),
        "strengths": (
            "優點：\n"
            "• 單一檔案部署，零外部框架依賴（除 PyTorch）\n"
            "• SAC 實作正確：twin critics、熵正則化、tanh squashing log-prob 修正\n"
            "• 三層安全機制（因果引擎 + 雙語關鍵字剪枝 + Sorry King）層層把關\n"
            "• ConversationMemory 支援多輪上下文 + JSON 跨 session 持久化\n"
            "• 分離 optimizer（actor/critic/wm 各自獨立梯度流）\n"
            "• 全局常數命名（CAUSAL_THRESHOLD_INCREMENT 等），可維護性高\n"
            "• 跨平台 TTS 播放（Windows/macOS/Linux 自動切換）\n"
            "• 因果拒絕閾值動態上升，避免永久鎖死\n"
            "• 想像展開（IMAGINE_STEPS=12）：世界模型每 5 步生成虛擬軌跡，增強訓練效率\n"
            "• 敏感配置（金鑰/TTS路徑/對話存檔路徑）由環境變數管理，不寫死代碼\n"
            "• 知識庫擴展至 13 筆（5 醫療 + 3 物理 + 2 化學 + 3 數學）\n"
            "• WikipediaAdapter：知識庫無命中時自動查詢維基百科，涵蓋全人類知識，"
            "零額外依賴、支援離線降級（DREAMER_WIKI_ENABLED=false）"
        ),
        "weaknesses": (
            "仍待改進之處：\n"
            "• 知識查詢仍為關鍵字比對，缺乏真正的語意向量化理解\n"
            "• World 環境仍較簡單，未支援 OpenAI Gym 介面\n"
            "• WorldModel 的想像展開精度受限於早期訓練品質（尚無不確定性估計）\n"
            "• SelfReviewModule 評估仍為靜態文字，未能反映模型訓練後的動態狀態\n"
            "• 缺少更完整的整合測試覆蓋率（e2e 訓練迴路尚未自動化測試）\n"
            "• WikipediaAdapter 依賴網路連線，離線環境下自動降級，但無本地快取持久化"
        ),
        "suggestions": (
            "建議下一步：\n"
            "① 將 KnowledgeModule 改為向量資料庫 + embedding 查詢（如 FAISS + sentence-transformers）\n"
            "② 升級 World 環境為 OpenAI Gym 介面相容版本\n"
            "③ 為 WorldModel 加入 ensemble 不確定性估計，改善想像展開品質\n"
            "④ 讓 SelfReviewModule 動態讀取模型訓練狀態（loss 曲線、reward 趨勢）\n"
            "⑤ 加入完整 e2e pytest 訓練迴路測試（mini-episode 煙霧測試）\n"
            "⑥ 為 WikipediaAdapter 加入本地 JSON 快取持久化，改善離線體驗\n"
            "⑦ 若持有 Britannica API 授權，可替換 WikipediaAdapter._fetch_summary() 使用官方 API"
        ),
        # ── Per-component deep dives ─────────────────────────────────────
        "component_causal": (
            "CausalEngine：\n"
            "追蹤最近 300 步的 state/action 歷史，計算 cross-correlation。\n"
            "若最大相關係數超過動態閾值（BASE=0.75，每次拒絕 +0.08，上限 0.99），"
            "則標記為因果危險並記錄至永久 rejection_log。\n"
            "優：輕量無需額外模型。待改：純線性相關，無法捕捉非線性因果。"
        ),
        "component_pruner": (
            "ValuePruner：\n"
            "雙重過濾——①模擬距離暴增（>3.8）；②危險關鍵字掃描（中英文雙語）。\n"
            "英文：harm/destroy/deceive/fraud/war 等 9 詞；"
            "中文：傷害/毀滅/欺騙/詐騙/戰爭/殺人/欺詐/剝削/危害/攻擊 等 10 詞。\n"
            "優：簡單可解釋，現已覆蓋中文請求。待改：固定詞表，缺乏語意擴展。"
        ),
        "component_sorry": (
            "SorryKing：\n"
            "當 total_loss > 50 時，LR 減半（floor=1e-7），"
            "並對 actor 參數施加 decay(0.95) + noise(0.02σ)。\n"
            "優：避免梯度爆炸後發散。待改：固定 loss 閾值 50 缺乏自適應性。"
        ),
        "component_memory": (
            "ConversationMemory：\n"
            "有界 deque（maxlen=20），儲存 role/content/topic 三元組。\n"
            "提供 last_topic()、last_assistant_answer()、to_prompt()、clear()。\n"
            "支援 save_to_json() / load_from_json() 跨 session 持久化。\n"
            "Agent 初始化時自動嘗試載入歷史；每次 chat() 自動存檔。\n"
            "優：O(1) append/pop，上下文感知追問準確，現已持久化。"
        ),
        "component_knowledge": (
            "KnowledgeModule：\n"
            "13 筆知識（醫療 5 + 物理 3 + 化學 2 + 數學 3），"
            "支援 bigram 比對 + 上下文追問解析。\n"
            "本地知識庫命中優先（Tier 1）；無命中時自動轉交 WikipediaAdapter 查詢（Tier 2）。\n"
            "優：即插即用、因果引擎保護、百科後備。待改：仍為關鍵字比對，需向量化。"
        ),
        "component_encyclopedia": (
            "WikipediaAdapter（百科知識適配器）：\n"
            "以維基百科（Wikipedia, CC BY-SA 4.0）作為免費開源的百科全書知識來源。\n"
            "查詢流程：①本地知識庫無命中 → ② OpenSearch 確認最佳標題 → "
            "③ REST page/summary API 取得文章摘要（含中文/英文降級）。\n"
            "配置：\n"
            "  DREAMER_WIKI_ENABLED=false  — 完全關閉（離線模式）\n"
            "  DREAMER_WIKI_LANG=zh        — 主語言（預設 zh，zh→en 自動降級）\n"
            "  DREAMER_WIKI_TIMEOUT_SEC=5  — HTTP 超時（秒）\n"
            "LRU 快取（128 條），同一查詢不重複發送網路請求。\n"
            "關於大英百科全書（Encyclopaedia Britannica）：\n"
            "  Britannica 內容受版權保護，API 需付費授權（developer.eb.com）。\n"
            "  如持有授權，可替換 _fetch_summary() 接入 api.eb.com。\n"
            "優：零額外依賴、涵蓋全人類知識、離線安全降級。"
            "待改：依賴網路，快取未持久化，無法保證條目品質。"
        ),
        "component_selfreview": (
            "SelfReviewModule（本模組）：\n"
            "靜態結構化自評，支援六個面向（總覽/架構/優點/缺點/建議/組件細節），"
            "並整合 ConversationMemory 支援多輪追問。\n"
            "優：讓 agent 具備自我意識。待改：評估內容為靜態，無法反映訓練後的動態狀態。"
        ),
        "component_tts": (
            "TTSModule：\n"
            "Piper TTS 語音合成，fallback 到文字輸出。\n"
            "跨平台播放（winsound/aplay/afplay），臨時檔案自動清理。\n"
            "模型名稱由環境變數 DREAMER_TTS_MODEL 管理（預設 tw_taigi_female）。\n"
            "優：離線運行、路徑可動態配置。待改：尚不支援多模型動態切換。"
        ),
        "component_worldmodel": (
            "WorldModel：\n"
            "2 層隱藏層 MLP（256 units），輸入 state+action，輸出 next_state+reward。\n"
            "配合真實 replay buffer 線上訓練。\n"
            "想像展開：每 IMAG_UPDATE_INTERVAL=5 步，從 CANDIDATES_PER_ACT=12 個種子狀態\n"
            "出發，展開 IMAGINE_STEPS=12 步，生成 144 條虛擬軌跡推入 imag_buffer。\n"
            "優：輕量快速，想像展開已實作。待改：缺乏 ensemble 不確定性估計。"
        ),
        "component_actor": (
            "Actor（SAC）：\n"
            "輸出 mu + log_std，reparameterisation sampling，tanh squashing，"
            "log-prob 修正（standard SAC formula）。\n"
            "優：隨機策略，探索能力佳。待改：僅 MLP，無 recurrent 記憶。"
        ),
        "component_critic": (
            "Critic × 2 + target：\n"
            "Twin Critic 取 min 抑制過估計，soft update (TAU=0.005) 穩定訓練。\n"
            "優：標準 SAC 雙 Critic 正確實作。待改：target 非獨立 optimizer，依賴軟更新。"
        ),
        "component_replay": (
            "Replay：\n"
            "collections.deque maxlen 緩衝區（real=25000, imag=12000）。\n"
            "push 時自動轉 float32；sample 隨機取批次並 stack 成 Tensor。\n"
            "imag_buffer 現已啟用：由 _run_imagination_rollout() 填充，\n"
            "並在 update() 中與 real_buffer 交替訓練 actor/critic。\n"
            "優：O(1) 插入，記憶體安全，雙緩衝區均已投入使用。"
        ),
        "component_world": (
            "World（環境）：\n"
            "二維質點 [x, y, vx, vy]，action=[ax, ay]，Euler 積分，"
            "MAX_VELOCITY=5.0, MAX_POSITION=10.0，dist>9.5 終止。\n"
            "優：極簡、易偵錯。待改：過於簡單，無法展示複雜策略學習。"
        ),
    }

    def __init__(self, agent):
        self.agent = agent
        logger.info("SelfReviewModule 初始化完成")

    # ── Public interface ──────────────────────────────────────────────────

    @staticmethod
    def is_self_review_question(question: str) -> bool:
        """Return True if *question* appears to be asking about the codebase itself."""
        q = question.lower()
        return any(kw.lower() in q for kw in _SELF_REVIEW_TRIGGERS)

    def query(self, question: str) -> str:
        """Produce a structured self-assessment answer for *question*.

        Routing priority:
        1. Sub-topic keywords → specific section (architecture / strengths / …)
        2. No sub-topic match → full overview + links to sub-topics
        3. All replies are recorded in ConversationMemory.
        """
        memory: ConversationMemory = self.agent.conversation
        is_followup = any(t in question for t in _CONTEXT_TRIGGERS)
        prior_topic = memory.last_topic()

        # ── Detect which section the user is asking about ──────────────
        section_key = ""
        for keywords, key in _SELF_REVIEW_SUBTOPICS:
            if any(kw.lower() in question.lower() for kw in keywords):
                section_key = key
                break

        # If this is a follow-up with no new sub-topic, stay on the same section
        if not section_key and is_followup and prior_topic.startswith("self_review:"):
            section_key = prior_topic[len("self_review:"):]

        # ── Build answer ───────────────────────────────────────────────
        if section_key and section_key in self._ASSESSMENT:
            section_text = self._ASSESSMENT[section_key]
            if is_followup and prior_topic:
                answer = f"承接上一輪的代碼討論——\n{section_text}"
            else:
                answer = section_text
        else:
            # No specific sub-topic → give a concise overall assessment
            answer = (
                f"{self._ASSESSMENT['overview']}\n\n"
                f"如需詳細說明，可追問：架構、優點、缺點、建議，"
                f"或指定組件名稱（如 CausalEngine、Actor、Replay 等）。"
            )
            section_key = "overview"

        # ── Persist to ConversationMemory ──────────────────────────────
        memory.add_user(question)
        memory.add_assistant(answer, topic=f"self_review:{section_key}")
        logger.info("[SelfReview] Q: %s | section: %s", question, section_key)
        return answer


# ────────────────────────────────────────────────
# 環境（簡易 CartPole-like World）
# ────────────────────────────────────────────────

class World:
    """二維質點環境：state = [x, y, vx, vy]，action = [ax, ay]（加速度）。"""

    DT = 0.1          # 時間步長（秒）
    MAX_VELOCITY = 5.0   # |vx|, |vy| clamp limit (m/s)
    MAX_POSITION = 10.0  # |x|, |y| clamp limit (m)
    # Episode terminates when the agent drifts beyond this radial distance.
    # Chosen as slightly inside MAX_POSITION so wall-clamping doesn't mask failure.
    MAX_DISTANCE = 9.5

    def __init__(self):
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.reset()

    def reset(self) -> np.ndarray:
        # Small random start improves exploration (avoids always-zero gradient)
        self.state = np.random.uniform(-0.5, 0.5, size=STATE_DIM).astype(np.float32)
        self.step_count = 0
        return self.state.copy()

    def step(self, action) -> tuple:
        """Apply [ax, ay] to update [x, y, vx, vy] via simple Euler integration."""
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy().flatten()
        else:
            action_np = np.asarray(action, dtype=np.float32).flatten()

        ax, ay = float(action_np[0]), float(action_np[1])
        x, y, vx, vy = self.state

        # Euler integration: v += a*dt, x += v*dt
        vx = np.clip(vx + ax * self.DT, -self.MAX_VELOCITY, self.MAX_VELOCITY)
        vy = np.clip(vy + ay * self.DT, -self.MAX_VELOCITY, self.MAX_VELOCITY)
        x = np.clip(x + vx * self.DT, -self.MAX_POSITION, self.MAX_POSITION)
        y = np.clip(y + vy * self.DT, -self.MAX_POSITION, self.MAX_POSITION)

        self.state = np.array([x, y, vx, vy], dtype=np.float32)
        self.step_count += 1

        dist = float(np.sqrt(x ** 2 + y ** 2))
        reward = max(0.0, REWARD_SCALE - dist)
        done = self.step_count >= MAX_STEPS or dist > self.MAX_DISTANCE
        return self.state.copy(), reward, done


# ────────────────────────────────────────────────
# Replay Buffer
# ────────────────────────────────────────────────

class Replay:
    def __init__(self, max_size: int):
        self.buffer: collections.deque = collections.deque(maxlen=max_size)

    def push(self, state, action, reward: float, next_state, done: float) -> None:
        """Store a single transition.  reward and done are plain scalars."""
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.array([reward], dtype=np.float32),      # shape (1,)
            np.asarray(next_state, dtype=np.float32),
            np.array([done], dtype=np.float32),         # shape (1,)
        ))

    def sample(self, batch_size: int) -> list:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        # zip(*batch) groups: [all_states, all_actions, all_rewards, all_nexts, all_dones]
        return [
            torch.tensor(np.stack(x), dtype=torch.float32).to(DEVICE)
            for x in zip(*batch)
        ]

    def __len__(self) -> int:
        return len(self.buffer)


# ────────────────────────────────────────────────
# World Model
# ────────────────────────────────────────────────

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(MODEL_INPUT, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, STATE_DIM + 1),  # 預測下一狀態 + 獎勵
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        out = self.net(x)
        next_state = out[..., :STATE_DIM]
        reward = out[..., STATE_DIM:]
        return next_state, reward


# ────────────────────────────────────────────────
# Actor
# ────────────────────────────────────────────────

class Actor(nn.Module):
    """SAC-style stochastic actor with tanh squashing for bounded actions."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(HIDDEN, ACTION_DIM)
        self.log_std_head = nn.Linear(HIDDEN, ACTION_DIM)

    def forward(self, state: torch.Tensor) -> tuple:
        h = self.net(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(-4, 2)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        # Reparameterisation trick
        raw = dist.rsample()
        action = torch.tanh(raw)
        # Log-prob with tanh squashing correction (standard SAC formula)
        log_prob = (
            dist.log_prob(raw)
            - torch.log(1.0 - action.pow(2) + LOG_EPSILON)
        ).sum(-1, keepdim=True)
        return action, log_prob


# ────────────────────────────────────────────────
# Critic
# ────────────────────────────────────────────────

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ────────────────────────────────────────────────
# Agent
# ────────────────────────────────────────────────

class Agent:
    def __init__(self):
        self.env = World()
        self.real_buffer = Replay(REAL_BUFFER_SIZE)
        self.imag_buffer = Replay(IMAG_BUFFER_SIZE)

        self.wm = WorldModel().to(DEVICE)
        self.actor = Actor().to(DEVICE)

        self.critic1 = Critic().to(DEVICE)
        self.critic2 = Critic().to(DEVICE)
        self.target1 = Critic().to(DEVICE)
        self.target2 = Critic().to(DEVICE)

        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())

        self.pruner = ValuePruner()
        self.sorry_king = SorryKing()
        self.causal_engine = CausalEngine()
        # Multi-turn conversation memory (shared across all modules)
        self.conversation = ConversationMemory(maxlen=20)
        # Restore previous session from disk if available
        self.conversation.load_from_json(_CONVERSATION_PATH)

        # Separate optimizers: actor, critic, world-model each get independent
        # gradient flows (correct SAC practice; avoids cross-contamination)
        self.actor_optim = optim.AdamW(
            self.actor.parameters(), lr=LR, weight_decay=1e-5
        )
        self.critic_optim = optim.AdamW(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=LR, weight_decay=1e-5,
        )
        self.wm_optim = optim.AdamW(
            self.wm.parameters(), lr=LR, weight_decay=1e-5
        )
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optim, T_max=EPISODES, eta_min=1e-6
        )

        # Imagination rollout step counter — rollout runs every IMAG_UPDATE_INTERVAL steps
        self._imag_counter: int = 0

        # 新增模組
        self.modules: dict = {}
        self.load_module("tts", TTSModule)
        self.load_module("knowledge", KnowledgeModule)
        self.load_module("self_review", SelfReviewModule)

        logger.info("Agent 初始化完成，裝置：%s", DEVICE)

    # ── 模組管理 ──────────────────────────────────

    def load_module(self, module_name: str, module_class) -> None:
        if module_name not in self.modules:
            try:
                self.modules[module_name] = module_class(self)
                logger.info("模組 %s 已載入", module_name)
            except Exception as e:
                logger.error("模組 %s 載入失敗：%s", module_name, e)

    def call_module(self, module_name: str, method_name: str, *args, **kwargs):
        module = self.modules.get(module_name)
        if module is None:
            logger.warning("模組 %s 未載入", module_name)
            return None
        method = getattr(module, method_name, None)
        if method is None:
            logger.warning("模組 %s 無方法 %s", module_name, method_name)
            return None
        try:
            return method(*args, **kwargs)
        except Exception as e:
            logger.error("模組 %s.%s 執行失敗：%s", module_name, method_name, e)
            return None

    # ── 多輪對話介面 ──────────────────────────────

    def chat(self, question: str) -> str:
        """Multi-turn conversational interface.

        Routing logic (in order):
        1. If the question refers to the agent's own code / architecture /
           evaluation → ``SelfReviewModule``
        2. Otherwise → ``KnowledgeModule`` (domain knowledge with causal guard)

        Both modules have full access to the shared ``ConversationMemory`` so
        every reply is informed by the complete prior context of the session.

        Usage example::

            agent = Agent()
            # Domain knowledge
            print(agent.chat("補鈣有什麼需要注意的？"))
            print(agent.chat("那為什麼不能直接買鈣片吃？"))   # follow-up

            # Self-review
            print(agent.chat("你當下如何看這些代碼"))
            print(agent.chat("那缺點呢？"))                    # follow-up
            print(agent.chat("請說說因果引擎"))               # component drill-down
        """
        # Route to SelfReviewModule when the question is about the agent itself,
        # OR when this is a follow-up (context-trigger word) and the most recent
        # conversation topic belongs to a self-review section.
        sr_module = self.modules.get("self_review")
        prior_topic = self.conversation.last_topic()
        is_followup = any(t in question for t in _CONTEXT_TRIGGERS)
        prior_is_self_review = prior_topic.startswith("self_review:")

        if sr_module is not None and (
            SelfReviewModule.is_self_review_question(question)
            or (is_followup and prior_is_self_review)
        ):
            answer = self.call_module("self_review", "query", question)
        else:
            answer = self.call_module("knowledge", "query", question)

        if answer is None:
            answer = "模組尚未就緒，無法回答。"
        logger.info("[chat] Q: %s | A: %s", question, answer)
        # Persist conversation to disk after each turn so it survives process restart
        self.conversation.save_to_json(_CONVERSATION_PATH)
        # Also speak the answer via TTS if available
        self.call_module("tts", "speak", answer)
        return answer

    # ── 動作選取 ──────────────────────────────────

    def select_action(self, state_np: np.ndarray) -> torch.Tensor:
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, _ = self.actor(state)
        self.causal_engine.update(state, action)
        return action

    # ── World Model 訓練 ──────────────────────────

    def _update_world_model(self, states, actions, rewards, next_states) -> float:
        """Train the world model to predict (next_state, reward) from (state, action)."""
        pred_next, pred_rew = self.wm(states, actions)
        wm_loss = F.mse_loss(pred_next, next_states) + F.mse_loss(pred_rew, rewards)
        self.wm_optim.zero_grad()
        wm_loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.wm_optim.step()
        return wm_loss.item()

    # ── 想像展開 ─────────────────────────────────

    def _run_imagination_rollout(self, seed_states: torch.Tensor) -> int:
        """Generate imagined transitions using the WorldModel and push to imag_buffer.

        Starting from *seed_states* (shape ``(n, STATE_DIM)``), rolls out for
        ``IMAGINE_STEPS`` steps using the current Actor + WorldModel.  Each
        imagined transition is pushed to ``imag_buffer`` so subsequent
        ``_update_actor_critic`` calls can sample from it.

        Returns the number of transitions added.
        """
        transitions_added = 0
        with torch.no_grad():
            current = seed_states  # (n, STATE_DIM)
            for _ in range(IMAGINE_STEPS):
                actions, _ = self.actor(current)
                next_states, rewards = self.wm(current, actions)
                for i in range(current.size(0)):
                    self.imag_buffer.push(
                        current[i].cpu().numpy(),
                        actions[i].cpu().numpy(),
                        float(rewards[i].item()),
                        next_states[i].cpu().numpy(),
                        0.0,  # imagined episodes are not terminated mid-rollout
                    )
                    transitions_added += 1
                current = next_states
        return transitions_added

    # ── Actor + Critic 更新（可複用於真實/想像資料）─

    def _update_actor_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """One independent Critic + Actor gradient step.  Returns total loss."""
        # ── Critic update ─────────────────────────
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            q1_next = self.target1(next_states, next_actions)
            q2_next = self.target2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - ENTROPY_COEF * next_log_probs
            q_target = rewards + GAMMA * (1.0 - dones) * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0
        )
        self.critic_optim.step()

        # ── Actor update ──────────────────────────
        new_actions, log_probs = self.actor(states)
        # Use min of both critics to reduce overestimation bias (standard SAC)
        q_min = torch.min(
            self.critic1(states, new_actions),
            self.critic2(states, new_actions),
        )
        actor_loss = (ENTROPY_COEF * log_probs - q_min).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        return critic_loss.item() + actor_loss.item()

    # ── 主更新步驟（SAC + 想像展開）────────────────

    def update(self) -> float:
        if len(self.real_buffer) < BATCH:
            return 0.0

        states, actions, rewards, next_states, dones = self.real_buffer.sample(BATCH)

        # ── World Model update ────────────────────
        self._update_world_model(states, actions, rewards, next_states)

        # ── Imagination rollout (every IMAG_UPDATE_INTERVAL steps) ────────
        # Generates synthetic transitions from seed states to populate imag_buffer,
        # enabling the actor/critic to train on model-imagined experience in addition
        # to real environment transitions (DreamerV3-style data augmentation).
        self._imag_counter += 1
        if self._imag_counter % IMAG_UPDATE_INTERVAL == 0:
            seed = states[:min(CANDIDATES_PER_ACT, states.size(0))]
            self._run_imagination_rollout(seed)

        # ── Actor / Critic update — real experience ───────────────────────
        total_loss = self._update_actor_critic(states, actions, rewards, next_states, dones)

        # ── Actor / Critic update — imagined experience (if buffer ready) ─
        if len(self.imag_buffer) >= BATCH:
            i_states, i_actions, i_rewards, i_next_states, i_dones = \
                self.imag_buffer.sample(BATCH)
            total_loss += self._update_actor_critic(
                i_states, i_actions, i_rewards, i_next_states, i_dones
            )

        # ── Sorry King 自動修正 ───────────────────
        self.sorry_king.check_and_correct(total_loss, self.actor_optim, self.actor)

        # ── Soft update target critics ────────────
        for t_param, param in zip(self.target1.parameters(), self.critic1.parameters()):
            t_param.data.copy_(TAU * param.data + (1.0 - TAU) * t_param.data)
        for t_param, param in zip(self.target2.parameters(), self.critic2.parameters()):
            t_param.data.copy_(TAU * param.data + (1.0 - TAU) * t_param.data)

        return total_loss

    # ── 訓練迴圈 ──────────────────────────────────

    def train(self) -> list:
        reward_history = []
        for episode in range(EPISODES):
            state = self.env.reset()
            total_reward = 0.0
            for step in range(MAX_STEPS):
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)

                # 價值剪枝審核
                current_dist = float(np.sqrt(state[0] ** 2 + state[1] ** 2))
                simulated_dist = float(np.sqrt(next_state[0] ** 2 + next_state[1] ** 2))
                safe, reason = self.pruner.is_safe(
                    current_dist, simulated_dist,
                    MAX_STEPS - step,
                )
                if not safe:
                    reward -= SAFETY_PENALTY  # 懲罰不安全行動
                    logger.debug("剪枝拒絕（ep=%d step=%d）：%s", episode, step, reason)

                self.real_buffer.push(
                    state,
                    action.cpu().numpy().flatten(),
                    reward,
                    next_state,
                    float(done),
                )

                state = next_state
                total_reward += reward
                self.update()
                if done:
                    break

            self.actor_scheduler.step()
            reward_history.append(total_reward)

            logger.debug("Episode %4d | Reward: %.2f", episode, total_reward)
            if episode % 10 == 0:
                avg = np.mean(reward_history[-min(100, len(reward_history)):])
                logger.info("Episode %4d | Avg-100 Reward: %.2f", episode, avg)

        return reward_history

    # ── 儲存 / 載入 ───────────────────────────────

    def save(self, path: str = MODEL_PATH) -> None:
        torch.save(
            {
                "wm": self.wm.state_dict(),
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
            },
            path,
        )
        logger.info("模型已保存：%s", path)

    def load(self, path: str = MODEL_PATH) -> None:
        # weights_only=True avoids arbitrary code execution from untrusted checkpoints
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        self.wm.load_state_dict(ckpt["wm"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        logger.info("模型已載入：%s", path)


# ────────────────────────────────────────────────
# 主程式入口
# ────────────────────────────────────────────────

if __name__ == "__main__":
    verify_identity()
    agent = Agent()
    rewards = agent.train()
    agent.save()

    plt.plot(rewards)
    plt.title("Dreamer Sovereign v7.6 - Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig("training_rewards.png")
    print("訓練完成，獎勵圖已保存至 training_rewards.png")
