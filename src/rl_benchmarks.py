import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Optional, Any
from src.config import (
    ACTOR_LR, BATCH_SIZE, GAMMA, TRAINING_EPISODES,
)
import warnings
warnings.filterwarnings('ignore')


class SoftmaxPortfolioWrapper(gym.Wrapper):
    """Wraps PortfolioEnv so that SB3 outputs raw logits that are softmax'd to valid weights."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        n_assets = env.action_space.shape[0]
        self.action_space = spaces.Box(
            low=-3.0, high=3.0, shape=(n_assets,), dtype=np.float32
        )

    def step(self, action: np.ndarray):
        e = np.exp(action - np.max(action))
        weights = (e / e.sum()).astype(np.float32)
        return self.env.step(weights)


class PortfolioCallback(BaseCallback):
    """Logs per-step portfolio returns during SB3 training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_returns: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'portfolio_return' in info:
                self.episode_returns.append(float(info['portfolio_return']))
        return True


def _resolve_timesteps(env: gym.Env) -> int:
    """Infer total training timesteps from env's returns attribute."""
    for obj in (env, getattr(env, 'unwrapped', None)):
        if obj is None:
            continue
        returns = getattr(obj, 'returns', None)
        if returns is not None:
            return len(returns) * TRAINING_EPISODES
    return 2520 * TRAINING_EPISODES  # fallback: ~10 years × 252 days


def train_ppo(
    env: gym.Env,
    total_timesteps: Optional[int] = None,
    seed: int = 42,
    n_assets: int = 20,
) -> PPO:
    if total_timesteps is None:
        total_timesteps = _resolve_timesteps(env)

    wrapped_env = SoftmaxPortfolioWrapper(env)
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
    )
    model = PPO(
        'MlpPolicy',
        wrapped_env,
        learning_rate=ACTOR_LR,
        n_steps=2048,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        seed=seed,
        verbose=0,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps)
    return model


def train_sac(
    env: gym.Env,
    total_timesteps: Optional[int] = None,
    seed: int = 42,
    n_assets: int = 20,
) -> SAC:
    if total_timesteps is None:
        total_timesteps = _resolve_timesteps(env)

    wrapped_env = SoftmaxPortfolioWrapper(env)
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256, 128], qf=[256, 256, 128]),
    )
    model = SAC(
        'MlpPolicy',
        wrapped_env,
        learning_rate=ACTOR_LR,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        seed=seed,
        verbose=0,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps)
    return model


def evaluate_sb3_agent(model, env: gym.Env, n_assets: int) -> Dict:
    """Run one deterministic evaluation episode; apply softmax to model outputs."""
    obs, _ = env.reset()
    portfolio_returns: List[float] = []
    all_weights: List[np.ndarray] = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        e = np.exp(action - np.max(action))
        weights = (e / e.sum()).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(weights)
        portfolio_returns.append(info['portfolio_return'])
        all_weights.append(info['weights'])

        if terminated or truncated:
            break

    return {'returns': portfolio_returns, 'weights': all_weights}
