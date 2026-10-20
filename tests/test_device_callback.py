import pytest
import torch as th
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from rl_experiments.builder import build_model
from rl_experiments.callbacks.device import HybridDeviceCallback, PhaseTimerCallback, move_optimizer_state
from rl_experiments.envs.factory import make_env

requires_cuda = pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA not available")

SMALL_PPO = {"n_steps": 32, "batch_size": 32, "n_epochs": 2}


class DeviceProbe(BaseCallback):
    """Record which device the acting policy is on during rollouts and after each update."""

    def __init__(self) -> None:
        super().__init__()
        self.rollout_devices: set[str] = set()
        self.rollout_end_devices: set[str] = set()

    def _on_step(self) -> bool:
        self.rollout_devices.add(str(self.model.policy.device.type))
        self.rollout_devices.add(str(self.model.device.type))
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_end_devices.add(str(self.model.policy.device.type))
        assert isinstance(self.model, OnPolicyAlgorithm)
        self.rollout_end_devices.add(str(self.model.rollout_buffer.device.type))


def _model(algo: str = "ppo", device: str = "cuda") -> OnPolicyAlgorithm:
    if algo == "ppo":
        env = make_env("CartPole-v1", 2, 0)
        return build_model("ppo", "MlpPolicy", env, hyperparams=SMALL_PPO, policy_kwargs={}, device=device, seed=0)
    env = make_env("CartPole-v1", 2, 0, mask_velocity=True)
    return build_model(
        "recurrent_ppo",
        "RecurrentPolicy",
        env,
        hyperparams=SMALL_PPO,
        policy_kwargs={"lstm_hidden_size": 8, "rnn_type": "gru"},
        device=device,
        seed=0,
    )


@requires_cuda
@pytest.mark.parametrize("strategy", ["move", "mirror"])
@pytest.mark.parametrize("algo", ["ppo", "recurrent_ppo"])
def test_hybrid_rollouts_on_cpu_training_on_gpu(strategy: str, algo: str) -> None:
    model = _model(algo)
    hybrid = HybridDeviceCallback("cpu", "cuda", strategy)  # type: ignore[arg-type]
    probe = DeviceProbe()
    # The probe runs after the hybrid callback, so on_rollout_end it sees the training device.
    model.learn(256, callback=CallbackList([hybrid, probe]))
    assert probe.rollout_devices == {"cpu"}
    assert probe.rollout_end_devices == {"cuda"}
    assert model.policy.device.type == "cuda"
    for state in model.policy.optimizer.state.values():
        assert state["exp_avg"].device.type == "cuda"
        assert state["step"].device.type == "cpu"
    assert hybrid.transfer_time > 0


@requires_cuda
def test_mirror_policy_receives_updated_weights() -> None:
    model = _model()
    hybrid = HybridDeviceCallback("cpu", "cuda", "mirror")
    model.learn(128, callback=hybrid)
    hybrid.on_rollout_start()  # sync once more, as the next iteration would
    assert hybrid._rollout_policy is not None
    assert hybrid._train_policy is not None
    for cpu_p, gpu_p in zip(hybrid._rollout_policy.parameters(), hybrid._train_policy.parameters(), strict=True):
        assert cpu_p.device.type == "cpu"
        th.testing.assert_close(cpu_p, gpu_p.cpu())


@requires_cuda
def test_hybrid_matches_cpu_learning() -> None:
    """Same seed → hybrid and pure-CPU runs should end with (numerically) close weights."""
    cpu_model = _model(device="cpu")
    cpu_model.learn(256)
    hybrid_model = _model(device="cuda")
    hybrid_model.learn(256, callback=HybridDeviceCallback("cpu", "cuda", "move"))
    for p_cpu, p_hyb in zip(cpu_model.policy.parameters(), hybrid_model.policy.parameters(), strict=True):
        th.testing.assert_close(p_cpu, p_hyb.cpu(), atol=1e-3, rtol=1e-3)


def test_move_optimizer_state_keeps_step_on_cpu() -> None:
    param = th.nn.Parameter(th.randn(3))
    optimizer = th.optim.Adam([param])
    param.grad = th.ones(3)
    optimizer.step()
    move_optimizer_state(optimizer, th.device("cpu"))
    state = optimizer.state[param]
    assert state["exp_avg"].device.type == "cpu"
    assert state["step"].device.type == "cpu"


def test_phase_timer_accounts_for_time() -> None:
    model = _model(device="cpu")
    timer = PhaseTimerCallback()
    model.learn(128, callback=timer)
    assert timer.rollout_time > 0
    assert timer.train_time > 0
