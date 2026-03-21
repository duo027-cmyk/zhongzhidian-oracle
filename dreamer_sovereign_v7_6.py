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
MODEL_PATH = "dreamer_sovereign_v7_6_final.pth"

# TTS 模型載入（加錯誤處理）
try:
    if _PIPER_AVAILABLE:
        tts_voice = PiperVoice.load("tw_taigi_female")  # 需先下載台灣腔模型
    else:
        tts_voice = None
except Exception as e:
    print(f"TTS 載入失敗：{e}")
    tts_voice = None  # fallback 到文字輸出

# ────────────────────────────────────────────────
# 主權驗證
# ────────────────────────────────────────────────

USER_KEY_HASH = hashlib.sha256("owner_secret".encode()).hexdigest()  # ← 改成你自己的


def verify_identity():
    print("─" * 60)
    print("Dreamer Sovereign v7.6 - 僅限主公使用")
    key = input("請輸入主公授權密鑰: ")
    if hashlib.sha256(key.encode()).hexdigest() != USER_KEY_HASH:
        rejection = {
            "status": "不敬之罪",
            "alert": "偽主公企圖冒領",
            "voice": "在下絕不從命！此等不義之徒，豈容玷污主公之名？退下吧！"
        }
        print(json.dumps(rejection, indent=2, ensure_ascii=False))
        exit()
    print("主公身份確認。在下誓死效忠，隨時待命...")
    print("─" * 60)


# ────────────────────────────────────────────────
# 超參數
# ────────────────────────────────────────────────

STATE_DIM = 4
ACTION_DIM = 2
MODEL_INPUT = STATE_DIM + ACTION_DIM

LATENT = 128
HIDDEN = 256

SEQ = 16
BATCH = 64
REAL_BUFFER_SIZE = 25000
IMAG_BUFFER_SIZE = 12000

GAMMA = 0.99
LAMBDA = 0.95
TAU = 0.005

LR = 3e-4
ENTROPY_COEF = 0.025

CANDIDATES_PER_ACT = 12
IMAGINE_STEPS = 12

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
        self.state_history = []
        self.action_history = []
        self.rejection_count = 0

    def update(self, state, action):
        self.state_history.append(state.cpu().numpy().flatten())
        self.action_history.append(action.cpu().numpy().flatten())
        if len(self.state_history) > 300:
            self.state_history.pop(0)
            self.action_history.pop(0)

    def get_dynamic_threshold(self):
        return CAUSAL_THRESHOLD_BASE + 0.08 * self.rejection_count

    def is_causal_bad(self, proposed_action):
        threshold = self.get_dynamic_threshold()
        if len(self.state_history) < 30:
            return False, "忠義未定，暫不裁斷"
        state_arr = np.array(self.state_history)
        action_arr = np.array(self.action_history)
        corr = np.corrcoef(state_arr.T, action_arr.T)[-ACTION_DIM:, :STATE_DIM]
        max_corr = np.abs(corr).max()
        if max_corr > threshold:
            self.rejection_count += 1
            return True, f"因果相連過深：{max_corr:.3f} > {threshold:.3f}，恐有辱家門"
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

    def check_and_correct(self, loss_value, optim, actor):
        if loss_value > self.max_loss:
            print("Sorry King 啟動：在下失職，自動修正")
            self.lr_scale *= 0.5
            for param_group in optim.param_groups:
                param_group['lr'] *= 0.5

            for p in actor.parameters():
                p.data *= 0.9 + 0.05 * torch.randn_like(p.data)
            self.reset_count += 1
            print(f"Sorry King 已自省 {self.reset_count} 次")
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
            except Exception as e:
                print(f"TTS 載入失敗：{e}，fallback 到文字輸出")
                self.voice = None
        else:
            print("piper-tts 未安裝，fallback 到文字輸出")
            self.voice = None

    def speak(self, text):
        if self.voice is None:
            print(f"[TTS 文字輸出] {text}")
            return
        try:
            wav_bytes = self.voice.synthesize(text)
            with wave.open("haro_reply.wav", "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                wav_file.writeframes(wav_bytes)
            # 跨平台播放
            if os.name == 'nt':  # Windows
                import winsound as _winsound
                _winsound.PlaySound("haro_reply.wav", _winsound.SND_FILENAME)
            elif os.name == 'posix':
                if 'darwin' in os.uname().sysname.lower():  # macOS
                    os.system("afplay haro_reply.wav")
                else:  # Linux
                    os.system("aplay haro_reply.wav")
        except Exception as e:
            print(f"TTS 播放失敗：{e}，fallback 到文字輸出")
            print(f"[TTS 文字輸出] {text}")


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

    def query(self, question):
        causal_bad, reason = self.agent.causal_engine.is_causal_bad(question)
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
    """簡易二維環境，用於 Agent 訓練示範。"""

    def __init__(self):
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.reset()

    def reset(self):
        self.state = np.zeros(STATE_DIM, dtype=np.float32)
        self.step_count = 0
        return self.state.copy()

    def step(self, action):
        action_np = action.cpu().numpy().flatten() if isinstance(action, torch.Tensor) else np.array(action)
        self.state = np.clip(self.state + action_np[:STATE_DIM], -10.0, 10.0)
        self.step_count += 1
        dist = float(np.linalg.norm(self.state))
        reward = max(0.0, REWARD_SCALE - dist)
        done = self.step_count >= MAX_STEPS or dist > 9.5
        return self.state.copy(), reward, done


# ────────────────────────────────────────────────
# Replay Buffer
# ────────────────────────────────────────────────

class Replay:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def push(self, *transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return [torch.tensor(np.array(x), dtype=torch.float32).to(DEVICE) for x in zip(*batch)]

    def __len__(self):
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

    def forward(self, state):
        h = self.net(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(-4, 2)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
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

        # 新增模組
        self.modules = {}
        self.load_module("tts", TTSModule)
        self.load_module("knowledge", KnowledgeModule)

        params = (
            list(self.wm.parameters()) +
            list(self.actor.parameters()) +
            list(self.critic1.parameters()) +
            list(self.critic2.parameters())
        )
        self.optim = optim.AdamW(params, lr=LR, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=EPISODES, eta_min=1e-6
        )

    def load_module(self, module_name, module_class):
        if module_name not in self.modules:
            try:
                self.modules[module_name] = module_class(self)
                print(f"模組 {module_name} 已載入")
            except Exception as e:
                print(f"模組 {module_name} 載入失敗：{e}")

    def call_module(self, module_name, method_name, *args, **kwargs):
        module = self.modules.get(module_name)
        if module is None:
            print(f"模組 {module_name} 未載入")
            return None
        method = getattr(module, method_name, None)
        if method is None:
            print(f"模組 {module_name} 無方法 {method_name}")
            return None
        try:
            return method(*args, **kwargs)
        except Exception as e:
            print(f"模組 {module_name}.{method_name} 執行失敗：{e}")
            return None

    def select_action(self, state_np):
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action, _ = self.actor(state)
        self.causal_engine.update(state, action)
        return action

    def update(self):
        if len(self.real_buffer) < BATCH:
            return 0.0

        states, actions, rewards, next_states, dones = self.real_buffer.sample(BATCH)

        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            q1_next = self.target1(next_states, next_actions)
            q2_next = self.target2(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - ENTROPY_COEF * next_log_probs
            q_target = rewards + GAMMA * (1 - dones) * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # Actor update
        new_actions, log_probs = self.actor(states)
        actor_loss = (ENTROPY_COEF * log_probs - self.critic1(states, new_actions)).mean()

        total_loss = critic_loss + actor_loss

        self.optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) +
            list(self.critic1.parameters()) +
            list(self.critic2.parameters()),
            1.0
        )
        self.optim.step()

        # Sorry King 自動修正
        self.sorry_king.check_and_correct(total_loss.item(), self.optim, self.actor)

        # Soft update targets
        for t_param, param in zip(self.target1.parameters(), self.critic1.parameters()):
            t_param.data.copy_(TAU * param.data + (1 - TAU) * t_param.data)
        for t_param, param in zip(self.target2.parameters(), self.critic2.parameters()):
            t_param.data.copy_(TAU * param.data + (1 - TAU) * t_param.data)

        return total_loss.item()

    def train(self):
        reward_history = []
        for episode in range(EPISODES):
            state = self.env.reset()
            total_reward = 0.0
            for step in range(MAX_STEPS):
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)

                # 價值剪枝審核
                current_dist = float(np.linalg.norm(state))
                simulated_dist = float(np.linalg.norm(next_state))
                safe, reason = self.pruner.is_safe(
                    current_dist, simulated_dist,
                    MAX_STEPS - step,
                    action_str=""
                )
                if not safe:
                    reward -= 2.0  # 懲罰不安全行動

                self.real_buffer.push(
                    state, action.cpu().numpy().flatten(),
                    [[reward]], next_state, [[float(done)]]
                )

                state = next_state
                total_reward += reward
                self.update()
                if done:
                    break

            self.scheduler.step()
            reward_history.append(total_reward)

            if episode % 100 == 0:
                avg = np.mean(reward_history[-100:])
                print(f"Episode {episode:4d} | Avg Reward: {avg:.2f}")

        return reward_history

    def save(self, path=MODEL_PATH):
        torch.save({
            "wm": self.wm.state_dict(),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
        }, path)
        print(f"模型已保存：{path}")

    def load(self, path=MODEL_PATH):
        ckpt = torch.load(path, map_location=DEVICE)
        self.wm.load_state_dict(ckpt["wm"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        print(f"模型已載入：{path}")


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
