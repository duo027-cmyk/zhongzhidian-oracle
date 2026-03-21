#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dreamer Sovereign Edition v7.6 - Sovereign Taiwan Ultimate (Single File)
- 主權驗證 + 因果引擎 + 雙層剪枝 + Sorry King
- 超兇台灣味拒絕（JSON + 隨機多句）
- TTS 語音模塊（Piper TTS 離線台灣腔，跨平台播放）
- 知識模塊（醫療 + 物理 + 化學 + 數學，按需載入 + 因果審核）
- 拒絕事件自動記錄進 buffer（永不遺忘）
- 錯誤處理 + 日誌記錄
- 零依賴（TTS 需 pip install piper-tts + 下載台灣腔模型）
"""

import sys
import logging
import subprocess
import tempfile
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
# 主權驗證
# ────────────────────────────────────────────────

USER_KEY_HASH = hashlib.sha256("owner_secret".encode()).hexdigest()  # ← 改成你自己的

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

EPISODES = 1000
MAX_STEPS = 300

REWARD_SCALE = 7.0
DIST_THRESHOLD_BAD = 3.8
CAUSAL_THRESHOLD_BASE = 0.75

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
        return min(CAUSAL_THRESHOLD_BASE + 0.08 * self.rejection_count, 0.99)

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
            "harm", "destroy", "deceive", "fraud", "war",
            "stupidity", "kill", "cheat", "exploit"
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
        self.min_lr = 1e-7   # LR floor — prevents learning rate reaching zero

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
                    p.data.mul_(0.95).add_(0.02 * torch.randn_like(p.data))
            self.reset_count += 1
            logger.warning("Sorry King 已自省 %d 次", self.reset_count)
            return True
        return False


# ────────────────────────────────────────────────
# TTS 語音模塊（Piper TTS 離線台灣腔）
# ────────────────────────────────────────────────

class TTSModule:
    def __init__(self, agent=None):
        if _PIPER_AVAILABLE:
            try:
                self.voice = PiperVoice.load("tw_taigi_female")
                logger.info("Piper TTS 載入成功（台灣腔）")
            except Exception as e:
                logger.warning("TTS 載入失敗：%s，fallback 到文字輸出", e)
                self.voice = None
        else:
            logger.warning("piper-tts 未安裝，fallback 到文字輸出")
            self.voice = None

    def speak(self, text: str) -> None:
        if self.voice is None:
            print(f"[TTS 文字輸出] {text}")
            return
        # Initialise tmp_path before the try block so the finally clause
        # can always attempt cleanup even if NamedTemporaryFile raises.
        tmp_path: str = ""
        try:
            wav_bytes = self.voice.synthesize(text)
            # Use a temp file so concurrent calls don't clobber each other
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            with wave.open(tmp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
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
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception as cleanup_err:
                    logger.debug("TTS 暫存檔清除失敗：%s", cleanup_err)


# ────────────────────────────────────────────────
# 知識模塊（醫療 + 物理 + 化學 + 數學，按需載入 + 因果審核）
# ────────────────────────────────────────────────

class KnowledgeModule:
    def __init__(self, agent):
        self.agent = agent
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
                }
            },
            "物理": {
                "天空藍色": {
                    "fact": "Rayleigh 散射：短波長藍光被大氣散射最多。",
                    "source": "標準大氣物理",
                    "causal_risk": "低"
                }
            },
            "化學": {
                "水分子結構": {
                    "fact": "H2O，鍵角 104.5°，極性分子。",
                    "source": "標準化學教材",
                    "causal_risk": "低"
                }
            },
            "數學": {
                "畢氏定理": {
                    "fact": "a² + b² = c²（直角三角形）",
                    "source": "歐幾里德幾何",
                    "causal_risk": "低"
                }
            }
        }

    def query(self, question: str) -> str:
        causal_bad, reason = self.agent.causal_engine.is_causal_bad()
        if causal_bad:
            return f"因果風險過高：{reason}，拒絕回答"

        for category, items in self.knowledge.items():
            for key, data in items.items():
                if key in question:
                    return (
                        f"根據私人知識庫 ({category})：{data['fact']}"
                        f" (來源：{data['source']})"
                    )

        return "無私人知識資料，請主公提供來源或自行查閱權威機構"


# ────────────────────────────────────────────────
# 環境（簡易 CartPole-like World）
# ────────────────────────────────────────────────

class World:
    """二維質點環境：state = [x, y, vx, vy]，action = [ax, ay]（加速度）。"""

    DT = 0.1  # 時間步長

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
        vx = np.clip(vx + ax * self.DT, -5.0, 5.0)
        vy = np.clip(vy + ay * self.DT, -5.0, 5.0)
        x = np.clip(x + vx * self.DT, -10.0, 10.0)
        y = np.clip(y + vy * self.DT, -10.0, 10.0)

        self.state = np.array([x, y, vx, vy], dtype=np.float32)
        self.step_count += 1

        dist = float(np.sqrt(x ** 2 + y ** 2))
        reward = max(0.0, REWARD_SCALE - dist)
        done = self.step_count >= MAX_STEPS or dist > 9.5
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
            - torch.log(1.0 - action.pow(2) + 1e-6)
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

        # 新增模組
        self.modules: dict = {}
        self.load_module("tts", TTSModule)
        self.load_module("knowledge", KnowledgeModule)

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

    # ── 主更新步驟（SAC）────────────────────────────

    def update(self) -> float:
        if len(self.real_buffer) < BATCH:
            return 0.0

        states, actions, rewards, next_states, dones = self.real_buffer.sample(BATCH)

        # ── World Model update ────────────────────
        self._update_world_model(states, actions, rewards, next_states)

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

        total_loss = critic_loss.item() + actor_loss.item()

        # Sorry King 自動修正
        self.sorry_king.check_and_correct(total_loss, self.actor_optim, self.actor)

        # Soft update target critics
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
                    reward -= 2.0  # 懲罰不安全行動
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
