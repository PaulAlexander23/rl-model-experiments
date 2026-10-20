import gymnasium as gym
import numpy as np
import pytest
import torch as th
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ConvertCallback

from rl_experiments.builder import build_model, parse_schedule
from rl_experiments.envs.factory import make_env
from rl_experiments.policies.extractors import MlpTrunkExtractor, TransformerWindowExtractor
from rl_experiments.policies.recurrent import GRUAsLSTM, RecurrentActorCriticPolicyV2

SMALL_PPO = {"n_steps": 16, "batch_size": 32, "n_epochs": 1}


def test_parse_schedule() -> None:
    assert parse_schedule(0.5) == 0.5
    schedule = parse_schedule("lin_0.002")
    assert callable(schedule)
    assert schedule(1.0) == pytest.approx(0.002)
    assert schedule(0.25) == pytest.approx(0.0005)
    with pytest.raises(ValueError, match="Unknown schedule"):
        parse_schedule("exp_0.1")


def test_shared_mlp_has_single_trunk() -> None:
    env = make_env("CartPole-v1", 2, 0)
    shared = build_model(
        "ppo",
        "MlpPolicy",
        env,
        hyperparams=SMALL_PPO,
        policy_kwargs={
            "features_extractor_class": "mlp_trunk",
            "features_extractor_kwargs": {"hidden_sizes": [64, 64]},
            "share_features_extractor": True,
            "net_arch": [],
        },
    )
    split = build_model(
        "ppo", "MlpPolicy", env, hyperparams=SMALL_PPO, policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
    )
    assert isinstance(shared.policy.features_extractor, MlpTrunkExtractor)
    assert shared.policy.pi_features_extractor is shared.policy.vf_features_extractor
    n_shared = sum(p.numel() for p in shared.policy.parameters())
    n_split = sum(p.numel() for p in split.policy.parameters())
    # Same hidden sizes, but split duplicates the trunk.
    assert n_split > 1.8 * n_shared
    shared.learn(64)


def test_transformer_extractor_output() -> None:
    space = gym.spaces.Box(-1, 1, shape=(3 * 5,))
    extractor = TransformerWindowExtractor(space, n_stack=5, d_model=16, n_heads=2, n_layers=1)
    out = extractor(th.randn(7, 15))
    assert out.shape == (7, 16)


def test_transformer_extractor_uses_all_frames() -> None:
    space = gym.spaces.Box(-1, 1, shape=(2 * 4,))
    extractor = TransformerWindowExtractor(space, n_stack=4, d_model=16, n_heads=2, n_layers=1)
    obs = th.randn(1, 8)
    changed = obs.clone()
    changed[0, 0] += 1.0  # oldest frame
    assert not th.allclose(extractor(obs), extractor(changed))


def test_gru_as_lstm_matches_gru() -> None:
    th.manual_seed(0)
    adapter = GRUAsLSTM(4, 8, num_layers=2)
    x = th.randn(5, 3, 4)
    h0, c0 = th.randn(2, 3, 8), th.randn(2, 3, 8)
    out, (h, c) = adapter(x, (h0, c0))
    ref_out, ref_h = adapter.gru(x, h0)
    th.testing.assert_close(out, ref_out)
    th.testing.assert_close(h, ref_h)
    assert c is c0


def _recurrent_model(**policy_kwargs: object) -> RecurrentPPO:
    env = make_env("CartPole-v1", 2, 0, mask_velocity=True)
    model = build_model(
        "recurrent_ppo",
        "RecurrentPolicy",
        env,
        hyperparams=SMALL_PPO,
        policy_kwargs={"lstm_hidden_size": 8, **policy_kwargs},
        seed=0,
    )
    assert isinstance(model, RecurrentPPO)
    return model


def test_gru_policy_replaces_lstm_and_optimizer() -> None:
    model = _recurrent_model(rnn_type="gru")
    policy = model.policy
    assert isinstance(policy, RecurrentActorCriticPolicyV2)
    assert isinstance(policy.lstm_actor, GRUAsLSTM)
    assert isinstance(policy.lstm_critic, GRUAsLSTM)
    optimized = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in policy.parameters()}
    model.learn(64)


def test_unknown_rnn_type() -> None:
    with pytest.raises(ValueError, match="Unknown rnn_type"):
        _recurrent_model(rnn_type="mamba")


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_fast_sequence_matches_loop(rnn_type: str) -> None:
    """Fast path == sb3-contrib's per-step loop when resets only happen at t=0 of each sequence."""
    th.manual_seed(0)
    policy = _recurrent_model(rnn_type=rnn_type).policy
    assert isinstance(policy, RecurrentActorCriticPolicyV2)
    n_seq, length, dim = 3, 6, policy.features_dim
    features = th.randn(n_seq * length, dim)
    states = (th.randn(1, n_seq, 8), th.randn(1, n_seq, 8))
    starts = th.zeros(n_seq, length)
    starts[1, 0] = 1.0  # second sequence begins a new episode
    starts = starts.flatten()

    fast, _ = policy._process_sequence(features, states, starts, policy.lstm_actor)
    slow, _ = RecurrentActorCriticPolicy._process_sequence(features, states, starts, policy.lstm_actor)  # type: ignore[arg-type]
    th.testing.assert_close(fast, slow)


def _rnn_grad_norm(model: RecurrentPPO) -> float:
    """Gradient reaching the actor RNN from the value loss only."""
    policy = model.policy
    buffer = model.rollout_buffer
    assert isinstance(policy, RecurrentActorCriticPolicyV2)
    assert isinstance(buffer, RecurrentRolloutBuffer)
    assert model.env is not None
    callback = ConvertCallback(None)
    callback.init_callback(model)
    model.collect_rollouts(model.env, callback, buffer, n_rollout_steps=model.n_steps)
    batch = next(buffer.get(32))
    policy.zero_grad()
    values, _, _ = policy.evaluate_actions(batch.observations, batch.actions, batch.lstm_states, batch.episode_starts)
    values.sum().backward()
    return sum(float(p.grad.norm()) for p in policy.lstm_actor.parameters() if p.grad is not None)


def test_shared_rnn_grad_controls_value_gradient() -> None:
    detached = _recurrent_model(enable_critic_lstm=False, shared_lstm=True, shared_rnn_grad=False)
    detached._setup_learn(1000)
    assert _rnn_grad_norm(detached) == 0.0

    shared = _recurrent_model(enable_critic_lstm=False, shared_lstm=True, shared_rnn_grad=True)
    shared._setup_learn(1000)
    assert _rnn_grad_norm(shared) > 0.0


def test_recurrent_policy_save_load_roundtrip(tmp_path: object) -> None:
    model = _recurrent_model(rnn_type="gru", shared_lstm=True, enable_critic_lstm=False, shared_rnn_grad=True)
    path = f"{tmp_path}/model.zip"
    model.save(path)
    loaded = RecurrentPPO.load(path)
    policy = loaded.policy
    assert isinstance(policy, RecurrentActorCriticPolicyV2)
    assert policy.rnn_type == "gru"
    assert policy.shared_rnn_grad
    obs = np.zeros((1, 2), dtype=np.float32)
    action, _ = loaded.predict(obs, deterministic=True)
    assert action.shape == (1,)


def test_ppo_is_used_for_mlp() -> None:
    model = build_model("ppo", "MlpPolicy", make_env("CartPole-v1", 1, 0), hyperparams=SMALL_PPO, policy_kwargs={})
    assert isinstance(model, PPO)
