"""Recurrent actor-critic policy for sb3-contrib's `RecurrentPPO`, extended with:

* `rnn_type="gru"`: a GRU wrapped to look like an `nn.LSTM` (sb3-contrib hard-codes LSTM (h, c) states).
  The cell state is carried through untouched, so the buffers still store it (wasted but harmless).
* `shared_rnn_grad`: sb3-contrib's `shared_lstm=True` *detaches* the critic from the LSTM, so the value
  loss never trains the recurrent layer. With `shared_rnn_grad=True` the value gradient flows into it.
* A fast path for `_process_sequence`: `RecurrentRolloutBuffer` splits sequences at episode starts, so
  during training a reset can only happen at t=0 of each padded sequence. Applying that reset to the
  initial state lets cuDNN process the whole sequence in one call instead of a Python loop per timestep.
"""

from __future__ import annotations

from typing import Any

import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from torch import nn


class GRUAsLSTM(nn.Module):
    """`nn.GRU` with the `nn.LSTM` call signature: `forward(x, (h, c)) -> (out, (h, c))`."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(
        self, x: th.Tensor, states: tuple[th.Tensor, th.Tensor]
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        hidden, cell = states
        out, hidden = self.gru(x, hidden.contiguous())
        return out, (hidden, cell)


class RecurrentActorCriticPolicyV2(RecurrentActorCriticPolicy):
    def __init__(
        self,
        *args: Any,
        rnn_type: str = "lstm",
        shared_rnn_grad: bool = False,
        fast_sequence: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rnn_type = rnn_type
        self.shared_rnn_grad = shared_rnn_grad
        self.fast_sequence = fast_sequence
        if rnn_type == "gru":
            self.lstm_actor = self._make_gru(self.lstm_actor)
            if self.lstm_critic is not None:
                self.lstm_critic = self._make_gru(self.lstm_critic)
            # The optimizer was built over the LSTM parameters; rebuild it over the GRU ones.
            lr_schedule = args[2] if len(args) > 2 else kwargs["lr_schedule"]
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]
        elif rnn_type != "lstm":
            raise ValueError(f"Unknown rnn_type {rnn_type!r}")

    def _make_gru(self, lstm: nn.Module) -> GRUAsLSTM:
        assert isinstance(lstm, nn.LSTM)
        return GRUAsLSTM(lstm.input_size, lstm.hidden_size, lstm.num_layers, **self.lstm_kwargs).to(self.device)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(rnn_type=self.rnn_type, shared_rnn_grad=self.shared_rnn_grad, fast_sequence=self.fast_sequence)
        return data

    def _process_sequence(  # type: ignore[override]
        self,
        features: th.Tensor,
        lstm_states: tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        lstm: nn.LSTM | GRUAsLSTM,
    ) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        if not self.fast_sequence:
            return RecurrentActorCriticPolicy._process_sequence(features, lstm_states, episode_starts, lstm)  # type: ignore[arg-type]
        n_seq = lstm_states[0].shape[1]
        sequence = features.reshape((n_seq, -1, lstm.input_size)).swapaxes(0, 1)
        starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)
        if not th.all(starts[1:] == 0.0):
            return RecurrentActorCriticPolicy._process_sequence(features, lstm_states, episode_starts, lstm)  # type: ignore[arg-type]
        keep = (1.0 - starts[0]).view(1, n_seq, 1)
        output, states = lstm(sequence, (keep * lstm_states[0], keep * lstm_states[1]))
        return th.flatten(output.transpose(0, 1), start_dim=0, end_dim=1), states

    def evaluate_actions(  # type: ignore[override]
        self, obs: th.Tensor, actions: th.Tensor, lstm_states: RNNStates, episode_starts: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        if not (self.shared_lstm and self.shared_rnn_grad):
            return super().evaluate_actions(obs, actions, lstm_states, episode_starts)
        # Same as the parent's shared-LSTM branch, minus the `.detach()` on the critic latent.
        features = self.extract_features(obs)
        assert isinstance(features, th.Tensor)
        pi_states = (lstm_states.pi[0], lstm_states.pi[1])
        latent, _ = self._process_sequence(features, pi_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent)
        latent_vf = self.mlp_extractor.forward_critic(latent)
        distribution = self._get_action_dist_from_latent(latent_pi)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        assert entropy is not None
        return values, distribution.log_prob(actions), entropy
