import pytest
import numpy as np
import torch
from src.ddpg import OUNoise, ReplayBuffer, Actor, Critic, DDPGAgent
from src.config import *

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fill_buffer(agent: DDPGAgent, state_dim: int, n_assets: int, n: int = 100):
    """Push n synthetic transitions into agent's replay buffer."""
    rng = np.random.default_rng(seed=0)
    weights = np.ones(n_assets) / n_assets
    for _ in range(n):
        state = rng.standard_normal(state_dim)
        next_state = rng.standard_normal(state_dim)
        reward = float(rng.standard_normal())
        agent.store_transition(state, weights, reward, next_state, False)


# ──────────────────────────────────────────────────────────────────────────────
# Actor tests
# ──────────────────────────────────────────────────────────────────────────────

def test_actor_output_is_simplex():
    """Actor output must be non-negative and sum to 1 (probability simplex)."""
    torch.manual_seed(0)
    actor = Actor(state_dim=20, n_assets=5)
    state = torch.randn(8, 20)
    weights = actor(state)

    assert weights.shape == (8, 5)
    assert (weights >= 0).all(), "All portfolio weights must be non-negative"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(8), atol=1e-6), (
        "Portfolio weights must sum to 1 for every sample"
    )


def test_actor_architecture():
    """Actor linear layers must have the exact shapes from paper Table 4."""
    state_dim, n_assets = 20, 5
    actor = Actor(state_dim=state_dim, n_assets=n_assets)
    params = dict(actor.named_parameters())

    # Sequential indices: 0=Linear, 1=ReLU, 2=Linear, 3=ReLU, 4=Linear, 5=ReLU, 6=Linear
    assert params["net.0.weight"].shape == (256, state_dim), "Layer-1 weight mismatch"
    assert params["net.0.bias"].shape   == (256,),           "Layer-1 bias mismatch"
    assert params["net.2.weight"].shape == (256, 256),       "Layer-2 weight mismatch"
    assert params["net.2.bias"].shape   == (256,),           "Layer-2 bias mismatch"
    assert params["net.4.weight"].shape == (128, 256),       "Layer-3 weight mismatch"
    assert params["net.4.bias"].shape   == (128,),           "Layer-3 bias mismatch"
    assert params["net.6.weight"].shape == (n_assets, 128),  "Output weight mismatch"
    assert params["net.6.bias"].shape   == (n_assets,),      "Output bias mismatch"


# ──────────────────────────────────────────────────────────────────────────────
# Critic tests
# ──────────────────────────────────────────────────────────────────────────────

def test_critic_output_shape():
    """Critic must return a (batch, 1) Q-value tensor."""
    torch.manual_seed(0)
    state_dim, n_assets, batch = 20, 5, 8
    critic = Critic(state_dim=state_dim, n_assets=n_assets)
    state = torch.randn(batch, state_dim)
    action = torch.randn(batch, n_assets)
    q = critic(state, action)

    assert q.shape == (batch, 1), f"Expected Q shape (8, 1), got {q.shape}"


# ──────────────────────────────────────────────────────────────────────────────
# OUNoise tests
# ──────────────────────────────────────────────────────────────────────────────

def test_ou_noise_properties():
    """OUNoise must produce correctly shaped samples and reset to zero state."""
    n_actions = 7
    noise = OUNoise(n_actions)

    sample = noise.sample()
    assert sample.shape == (n_actions,), f"Expected shape ({n_actions},), got {sample.shape}"
    assert isinstance(sample, np.ndarray), "Sample must be a numpy array"

    # Evolve state over several steps
    for _ in range(20):
        noise.sample()

    # After reset, internal state must be zeros
    noise.reset()
    assert np.array_equal(noise.state, np.zeros(n_actions)), "State must be zero after reset"


# ──────────────────────────────────────────────────────────────────────────────
# ReplayBuffer tests
# ──────────────────────────────────────────────────────────────────────────────

def test_replay_buffer_capacity():
    """Circular buffer must discard oldest entries once capacity is reached."""
    capacity = 100
    buf = ReplayBuffer(capacity=capacity)
    state_dim, n_assets = 10, 3
    rng = np.random.default_rng(seed=1)
    weights = np.ones(n_assets) / n_assets

    for _ in range(200):
        buf.push(
            rng.standard_normal(state_dim),
            weights,
            float(rng.standard_normal()),
            rng.standard_normal(state_dim),
            False,
        )

    assert len(buf) == capacity, f"Expected len {capacity}, got {len(buf)}"


def test_replay_buffer_sample():
    """Sampled batch must return correctly shaped FloatTensors."""
    state_dim, n_assets = 10, 3
    batch_size = 32
    buf = ReplayBuffer(capacity=200)
    rng = np.random.default_rng(seed=2)
    weights = np.ones(n_assets) / n_assets

    for _ in range(100):
        buf.push(
            rng.standard_normal(state_dim),
            weights,
            float(rng.standard_normal()),
            rng.standard_normal(state_dim),
            False,
        )

    states, actions, rewards, next_states, dones = buf.sample(batch_size)

    assert states.shape     == (batch_size, state_dim), f"states shape: {states.shape}"
    assert actions.shape    == (batch_size, n_assets),  f"actions shape: {actions.shape}"
    assert rewards.shape    == (batch_size, 1),         f"rewards shape: {rewards.shape}"
    assert next_states.shape== (batch_size, state_dim), f"next_states shape: {next_states.shape}"
    assert dones.shape      == (batch_size, 1),         f"dones shape: {dones.shape}"

    for tensor in (states, actions, rewards, next_states, dones):
        assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"


# ──────────────────────────────────────────────────────────────────────────────
# DDPGAgent tests
# ──────────────────────────────────────────────────────────────────────────────

def test_ddpg_agent_select_action():
    """Agent actions must lie on the probability simplex, with and without noise."""
    state_dim, n_assets = 10, 4
    agent = DDPGAgent(state_dim=state_dim, n_assets=n_assets, seed=0)
    rng = np.random.default_rng(seed=3)
    state = rng.standard_normal(state_dim)

    for add_noise in (False, True):
        action = agent.select_action(state, add_noise=add_noise)

        assert action.shape == (n_assets,), f"action shape mismatch (add_noise={add_noise})"
        assert (action >= 0).all(), f"Negative weight detected (add_noise={add_noise})"
        assert abs(action.sum() - 1.0) < 1e-5, (
            f"Weights sum to {action.sum():.6f} ≠ 1 (add_noise={add_noise})"
        )


def test_ddpg_agent_update_returns_losses():
    """update() must return (critic_loss, actor_loss) floats once buffer is full."""
    state_dim, n_assets = 10, 3
    agent = DDPGAgent(state_dim=state_dim, n_assets=n_assets, batch_size=64, seed=7)

    # Buffer needs >= batch_size transitions to trigger an update
    _fill_buffer(agent, state_dim, n_assets, n=100)

    result = agent.update()

    assert result is not None, "update() returned None despite sufficient buffer"
    c_loss, a_loss = result
    assert isinstance(c_loss, float), f"critic_loss must be float, got {type(c_loss)}"
    assert isinstance(a_loss, float), f"actor_loss must be float, got {type(a_loss)}"
    assert np.isfinite(c_loss), f"critic_loss is not finite: {c_loss}"
    assert np.isfinite(a_loss), f"actor_loss is not finite: {a_loss}"


def test_soft_update():
    """Target network must lag behind online network after a gradient update."""
    state_dim, n_assets = 10, 3
    agent = DDPGAgent(state_dim=state_dim, n_assets=n_assets, batch_size=64, seed=11)

    # Capture initial online weights (target is identical at init)
    initial_online = [p.clone().detach() for p in agent.actor.parameters()]

    _fill_buffer(agent, state_dim, n_assets, n=100)
    agent.update()

    online_params = list(agent.actor.parameters())
    target_params = list(agent.target_actor.parameters())

    # Online weights must have been updated by gradient descent
    online_changed = any(
        not torch.equal(p_new.detach(), p_old)
        for p_new, p_old in zip(online_params, initial_online)
    )
    assert online_changed, "Online actor weights must change after gradient update"

    # Target weights must differ from online weights (tau=0.001 << 1 → target lags)
    target_differs = any(
        not torch.equal(p_t.detach(), p_o.detach())
        for p_t, p_o in zip(target_params, online_params)
    )
    assert target_differs, (
        "Target and online actor networks must differ after soft update (tau < 1)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
