"""Actor, critic and observation normalizer for the asymmetric-critic PPO core.

Kept deliberately tiny and export-friendly: the actor is a plain MLP with tanh hidden
activations and a linear output, because it will later be mirrored by hand into about 80
lines of C (see `brick_builder/export/`). No
squashing function is baked into the network; the caller clips the sampled action to the
environment's bounds, which keeps the exported C actor a straight matmul/tanh chain.

The critic is privileged (asymmetric actor-critic, Pinto 2018): it is a separate network
that only ever sees the "critic" half of a Dict observation and is thrown away at export
time along with its normalizer.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _layer_init(layer: nn.Linear, gain: float, bias: float = 0.0) -> nn.Linear:
    """Orthogonal init (CleanRL-style). Keeps early activations well scaled, which matters
    a lot for converging within the short step budgets the acceptance tests use."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


def _mlp(sizes: Sequence[int], output_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers.append(_layer_init(nn.Linear(sizes[i], sizes[i + 1]), gain=2**0.5))
        layers.append(nn.Tanh())
    layers.append(_layer_init(nn.Linear(sizes[-2], sizes[-1]), gain=output_gain))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Gaussian policy over a continuous action space.

    `forward` returns the mean action only; `log_std` is a single learned vector shared
    across all observations (state-independent, per spec section 6), not a function of the
    input. This halves the export cost and matches the residual-primitive action model,
    where exploration noise should not depend on a possibly out-of-distribution input.

    Observations passed in must already be normalized (see `RunningNorm`); this class does
    not know about normalization so the frozen normalizer can be exported separately.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: Sequence[int] = (64, 64),
        init_log_std: float = -1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = _mlp((obs_dim, *hidden, act_dim), output_gain=0.01)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean action for a batch of normalized actor observations, shape (B, obs_dim)."""
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        """The Gaussian action distribution for a batch of normalized observations."""
        mean = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


class Critic(nn.Module):
    """State-value function over the privileged ("critic") observation.

    Never exported: it exists only to reduce the variance of the actor's PPO gradient
    during training. Larger than the actor (256x256 vs 64x64) because it has no runtime
    budget to respect.
    """

    def __init__(self, obs_dim: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = _mlp((obs_dim, *hidden, 1), output_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Value estimate, shape (B, 1), for a batch of normalized critic observations."""
        return self.net(obs)


class RunningNorm(nn.Module):
    """Running mean/variance normalizer (Welford's parallel algorithm, as in OpenAI
    Baselines' `RunningMeanStd`).

    Mean, variance and sample count are registered buffers, so `state_dict()` /
    `load_state_dict()` (inherited from `nn.Module`) round-trip them exactly through
    `torch.save`/`torch.load`, and a checkpointed actor carries its own frozen copy.

    Set `.frozen = True` (e.g. right before export, or during evaluation) to stop `update`
    from changing the statistics while `normalize` keeps working; the actor's exported C
    code always uses frozen mean/var baked in as constants.
    """

    def __init__(self, dim: int, eps: float = 1e-8, clip: float = 10.0):
        super().__init__()
        self.eps = eps
        self.clip = clip
        self.frozen = False
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Fold a batch `x` of shape (B, dim) into the running statistics. No-op if frozen."""
        if self.frozen:
            return
        x = torch.as_tensor(x, dtype=torch.float32)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot_count)
        self.count.copy_(tot_count)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """(x - mean) / sqrt(var + eps), clipped to +-`clip`. Does not update statistics."""
        x = torch.as_tensor(x, dtype=torch.float32)
        normed = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(normed, -self.clip, self.clip)
