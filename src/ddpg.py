import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, List, Optional
import random
from src.config import *


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""

    def __init__(self, n_actions: int, theta: float = OU_THETA, sigma: float = OU_SIGMA, mu: float = 0.0):
        self.n_actions = n_actions
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.state = np.zeros(n_actions)

    def reset(self):
        self.state = np.zeros(self.n_actions)

    def sample(self) -> np.ndarray:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)
        self.state = x + dx
        return self.state.copy()


class ReplayBuffer:
    """Circular experience replay buffer."""

    def __init__(self, capacity: int = REPLAY_BUFFER):
        self.memory: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, float(reward), next_state, float(done)))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.memory)


class Actor(nn.Module):
    """
    Policy network (Table 4): maps state -> softmax portfolio weights.

    Architecture:
        Dense(state_dim, 256, ReLU) -> Dense(256, 256, ReLU)
        -> Dense(256, 128, ReLU) -> Dense(128, n_assets) -> Softmax
    """

    def __init__(self, state_dim: int, n_assets: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_assets),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(state), dim=-1)


class Critic(nn.Module):
    """
    Value network (Table 5): maps (state, action) -> Q-value.

    Architecture:
        State branch  : Dense(state_dim, 256, ReLU) -> Dense(256, 128, ReLU)
        Action branch : Dense(n_assets, 128, ReLU)
        Concat 256-dim -> Dense(256, 128, ReLU) -> Dense(128, 1)
    """

    def __init__(self, state_dim: int, n_assets: int):
        super().__init__()
        self.state_branch = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.action_branch = nn.Sequential(
            nn.Linear(n_assets, 128),
            nn.ReLU(),
        )
        self.hidden = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.output = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        s = self.state_branch(state)
        a = self.action_branch(action)
        x = torch.cat([s, a], dim=-1)
        x = self.hidden(x)
        return self.output(x)


class DDPGAgent:
    """Deep Deterministic Policy Gradient agent for portfolio allocation."""

    def __init__(
        self,
        state_dim: int,
        n_assets: int,
        actor_lr: float = ACTOR_LR,
        critic_lr: float = CRITIC_LR,
        gamma: float = GAMMA,
        tau: float = TAU,
        buffer_capacity: int = REPLAY_BUFFER,
        batch_size: int = BATCH_SIZE,
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Online networks
        self.actor = Actor(state_dim, n_assets)
        self.critic = Critic(state_dim, n_assets)

        # Target networks initialised with same weights
        self.target_actor = Actor(state_dim, n_assets)
        self.target_critic = Critic(state_dim, n_assets)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Freeze target network gradient computation
        for p in self.target_actor.parameters():
            p.requires_grad = False
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.noise = OUNoise(n_assets)
        self.buffer = ReplayBuffer(buffer_capacity)

    # ------------------------------------------------------------------
    # Core agent methods
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).numpy()
        if add_noise:
            noise = self.noise.sample()
            action = action + noise
            # Re-normalise to a valid probability simplex via softmax
            action = np.exp(action - action.max())
            action = action / action.sum()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def _soft_update(self, online: nn.Module, target: nn.Module):
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.data.copy_(self.tau * p_online.data + (1.0 - self.tau) * p_target.data)

    def update(self) -> Optional[Tuple[float, float]]:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # ── Critic update ──────────────────────────────────────────────
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            y = rewards + self.gamma * target_q * (1.0 - dones)

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor update (maximize Q → minimize -Q) ───────────────────
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Soft-update target networks ───────────────────────────────
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic, self.target_critic)

        return critic_loss.item(), actor_loss.item()

    # ------------------------------------------------------------------
    # Episode-level helpers
    # ------------------------------------------------------------------

    def train_episode(self, env, episode_idx: int = 0) -> dict:
        """Run one full training episode; return summary statistics.

        Compatible with Gymnasium's 5-tuple step API and 2-tuple reset API.
        """
        obs = env.reset()
        # Gymnasium reset() returns (obs, info); handle both old and new API.
        if isinstance(obs, tuple):
            state = obs[0]
        else:
            state = obs
        self.noise.reset()

        total_reward = 0.0
        critic_losses: List[float] = []
        actor_losses: List[float] = []
        n_steps = 0
        done = False

        while not done:
            action = self.select_action(state, add_noise=True)
            step_result = env.step(action)
            # Gymnasium 5-tuple: (obs, reward, terminated, truncated, info)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            self.store_transition(state, action, reward, next_state, done)

            result = self.update()
            if result is not None:
                c_loss, a_loss = result
                critic_losses.append(c_loss)
                actor_losses.append(a_loss)

            state = next_state
            total_reward += float(reward)
            n_steps += 1

        return {
            "total_reward": total_reward,
            "critic_losses": critic_losses,
            "actor_losses": actor_losses,
            "n_steps": n_steps,
        }

    def evaluate(self, env) -> dict:
        """Run one episode without exploration noise; return evaluation results.

        Returns portfolio_returns (from info dict) rather than composite rewards,
        so callers can compute clean investment performance metrics.
        """
        obs = env.reset()
        if isinstance(obs, tuple):
            state = obs[0]
        else:
            state = obs

        total_reward = 0.0
        portfolio_returns: List[float] = []
        weights: List[np.ndarray] = []
        done = False

        while not done:
            action = self.select_action(state, add_noise=False)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            # Use actual portfolio return from info, not the composite reward
            port_ret = info.get("portfolio_return", float(reward))
            portfolio_returns.append(float(port_ret))
            weights.append(action.copy())
            state = next_state
            total_reward += float(reward)

        return {
            "returns": portfolio_returns,
            "weights": weights,
            "total_reward": total_reward,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_actor.load_state_dict(checkpoint["actor"])
        self.target_critic.load_state_dict(checkpoint["critic"])
