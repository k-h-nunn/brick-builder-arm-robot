"""The policy: pixels to a tool correction, differentiable all the way.

    256x256 wrist frame
      -> SceneModelNet                       (the shipped convolutional scene model's weights)
      -> slot_head.soft_masks/soft_blob_rows (soft assignment, K x W slot rows)
      -> soft match against the target rows  (observed minus target, the difference `G` reads)
      -> a `G`-shaped tanh MLP               (slot-row differences + mechanical state -> dx, dy, spin)
      -> residual

The encoder starts from `perception/weights/vision_model.pt` and is kept in evaluation mode so its
batch-norm statistics never move: a minibatch of thirty-two frames is far too small to re-estimate
them and letting it try loses the shipped model in the first update. It still receives gradients,
at a fraction of the head's learning rate (`train.encoder_lr_scale`).

Two things differ from the controller's own slate and are written down rather than glossed over.

- **Slot order.** The controller's slate is three named references plus the brick in the hand,
  ordered by title. A policy has no titles, so slots are filled in the frame's own candidate order
  -- the arm, the baseplate, then bricks by peak strength -- with no temporal assigner. That makes
  a forward pass a pure function of the frame, which is what lets an update replay exactly the
  pass the rollout took. Correspondence with the target state comes from the soft match below,
  which is by appearance and needs no shared ordering.
- **Width.** Because the slate is all `K = 8` slots rather than four, the MLP reads 97 numbers
  where `G` reads 54. It is `G`-shaped -- the same slot-row differences, the same mechanical
  state, the same two tanh layers, the same three outputs in the same units -- but the shipped
  `inverse.npz` weights do not fit it and are not loaded.

The output layer is initialised to zero, so an untrained policy's mean action is exactly zero and
the environment reproduces today's controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ...action.models.features import MECH_W, N_SLATE, ROW_FIELDS, ROW_W
from ...perception.cnn.model import SceneModelNet, unpack_weights
from ...perception.cnn.slot_head import SlotHeadConfig, soft_blob_rows, soft_masks
from ...perception.cnn.source import DEFAULT_WEIGHTS
from ...perception.scene_model import PLACEHOLDER_SLOT_SCHEMA
from ..nets import Critic, RunningNorm

__all__ = ["PolicyConfig", "VisionControlPolicy", "slot_rows", "match_target"]

K_SLOTS = 8
W_SLOT = PLACEHOLDER_SLOT_SCHEMA.W
PER_SLOT = 2 + ROW_W
"""Validity, weight, and the eight fields `models.features` reads -- the layout `encode_inverse`
builds, one entry per slot instead of one per slate position."""

PROGRESS_W = 3
POLICY_IN = K_SLOTS * PER_SLOT + MECH_W + 2 + PROGRESS_W

_READ = [PLACEHOLDER_SLOT_SCHEMA.fields[name] for name in ROW_FIELDS]
_MATCH = ["colour", "mass", "moments"]
"""What two rows are matched on: what the object *is*, not where it is. Matching on the centre
would pair the observed brick with whichever target object happens to sit where it sits."""


@dataclass(frozen=True)
class PolicyConfig:
    hidden: tuple[int, ...] = (64, 64)
    tau_match: float = 0.25
    """Width of the matching softmax, in the row's own units. Colour is in `[-1, 1]` per channel
    and two different bricks differ by order one, so a quarter is hard between colours and soft
    between two views of the same object."""
    init_log_std: float = -1.5
    """Starting exploration, in units of the residual's own saturation: about 0.22 of a full-scale
    residual per tick, which is 0.26 mm -- under the loop's own dead band, so exploration does not
    by itself push the tool outside what the analytic term would have commanded."""
    slot_cfg: SlotHeadConfig = SlotHeadConfig()
    weights: str | None = None


def slot_rows(net: SceneModelNet, image: torch.Tensor,
              cfg: SlotHeadConfig = SlotHeadConfig()) -> tuple[torch.Tensor, torch.Tensor]:
    """`[3, H, W]` frame in `[0, 1]` -> `([K, W]` rows, `[K]` validity), padded and truncated.

    Pure: no assigner, no state, so the same frame always gives the same rows.
    """
    out = net(image[None])
    masks, _ = soft_masks(out, cfg)
    rows = torch.zeros((K_SLOTS, W_SLOT))
    valid = torch.zeros(K_SLOTS)
    n = int(min(masks.shape[0], K_SLOTS))
    if n:
        built = soft_blob_rows(masks[:n], image, cfg)
        rows = torch.cat([built, torch.zeros((K_SLOTS - n, W_SLOT))], dim=0)
        valid = torch.cat([torch.ones(n), torch.zeros(K_SLOTS - n)])
    return rows, valid


def _identity(rows: torch.Tensor) -> torch.Tensor:
    """The fields two rows are matched on, stacked."""
    return torch.cat([PLACEHOLDER_SLOT_SCHEMA.get(rows, name) for name in _MATCH], dim=-1)


def match_target(rows: torch.Tensor, valid: torch.Tensor, target: torch.Tensor,
                 tau: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft one-to-many match of observed rows against the target state's rows.

    Returns `(matched target rows [K, W], match confidence [K])`. An empty target slot is masked
    out of the softmax rather than matched with a small weight, and a slot with nothing to match
    against comes back with confidence zero, which the feature builder turns into an invalid
    entry. Differentiable in the observed rows; the target rows are data.
    """
    tgt_present = (target.abs().sum(dim=1) > 0).float()
    cost = torch.cdist(_identity(rows)[None], _identity(target)[None])[0]      # [K, K]
    logits = -cost / tau + torch.log(tgt_present.clamp_min(1e-9))[None, :]
    weights = torch.softmax(logits, dim=1)
    matched = weights @ target
    confidence = valid * tgt_present.max().expand(valid.shape)
    return matched, confidence


def policy_features(rows: torch.Tensor, valid: torch.Tensor, target: torch.Tensor,
                    mech: torch.Tensor, progress: torch.Tensor, in_hand: float,
                    tau: float) -> torch.Tensor:
    """`[POLICY_IN]`: per-slot (validity, weight, eight differences), then the context.

    The same construction as `models.features.encode_inverse`, one entry per slot: an entry is
    differenced only where both sides have something, the weight is `1 - border fraction` as it is
    there, and the two context flags are "is there a brick in the hand" and "how many slots
    matched".
    """
    matched, confidence = match_target(rows, valid, target, tau)
    both = confidence.clamp(0.0, 1.0)
    diff = torch.cat([(rows - matched)[:, sl] for sl in _READ], dim=1) * both[:, None]
    weight = (1.0 - PLACEHOLDER_SLOT_SCHEMA.get(rows, "border")[:, 0]).clamp(0.0, 1.0) * both
    per_slot = torch.cat([both[:, None], weight[:, None], diff], dim=1)
    return torch.cat([per_slot.reshape(-1), mech[:MECH_W],
                      torch.stack([torch.as_tensor(float(in_hand)), both.sum()]),
                      progress[:PROGRESS_W]])


class VisionControlPolicy(nn.Module):
    """Encoder, slot head, `G`-shaped MLP, and the critic that is thrown away at export."""

    def __init__(self, cfg: PolicyConfig = PolicyConfig(), *, critic_dim: int,
                 hidden_critic: tuple[int, ...] = (256, 256)) -> None:
        super().__init__()
        self.cfg = cfg
        state = torch.load(cfg.weights or DEFAULT_WEIGHTS, map_location="cpu", weights_only=False)
        self.encoder = SceneModelNet(width=float(state.get("width", 1.0)))
        self.encoder.load_state_dict(unpack_weights(state["model"]))
        self.encoder.eval()

        sizes = (POLICY_IN, *cfg.hidden, 3)
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.head = nn.Sequential(*layers)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.log_std = nn.Parameter(torch.full((3,), float(cfg.init_log_std)))

        self.critic = Critic(critic_dim, hidden_critic)
        self.norm_critic = RunningNorm(critic_dim)
        self.norm_features = RunningNorm(POLICY_IN)

    def train(self, mode: bool = True):
        """Keep the encoder in evaluation mode whatever the rest of the module is doing.

        Batch normalization with a minibatch of a few dozen wrist frames would re-estimate
        statistics measured over a whole training set; the shipped weights would not survive it.
        The convolution weights still get gradients.
        """
        super().train(mode)
        self.encoder.eval()
        return self

    # -- one frame ---------------------------------------------------------------------------

    def forward_obs(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One actor observation -> `(features [POLICY_IN], rows [K, W], valid [K])`.

        One encoder pass answers both the policy's input and the alignment term's, so a training
        step never runs the scene model twice over the same frame.
        """
        image = _image(obs["image"])
        rows, valid = slot_rows(self.encoder, image, self.cfg.slot_cfg)
        target = torch.as_tensor(np.asarray(obs["target"]), dtype=torch.float32)
        mech = torch.as_tensor(np.asarray(obs["mech"]), dtype=torch.float32)
        progress = torch.as_tensor(np.asarray(obs["progress"]), dtype=torch.float32)
        features = policy_features(rows, valid, target, mech, progress,
                                   in_hand=1.0, tau=self.cfg.tau_match)
        return features, rows, valid

    def features(self, obs: dict) -> torch.Tensor:
        """`[POLICY_IN]` for one actor observation dict of numpy arrays or tensors."""
        return self.forward_obs(obs)[0]

    def slots(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """The policy's own scene state for one observation -- what the alignment term predicts."""
        return slot_rows(self.encoder, _image(obs["image"]), self.cfg.slot_cfg)

    def distribution(self, features: torch.Tensor) -> torch.distributions.Normal:
        mean = self.head(self.norm_features.normalize(features))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs_critic: torch.Tensor) -> torch.Tensor:
        return self.critic(self.norm_critic.normalize(obs_critic)).squeeze(-1)

    # -- parameter groups ----------------------------------------------------------------------

    def groups(self, lr: float, encoder_scale: float) -> list[dict]:
        """Optimizer groups: the encoder moves slowly, everything else at the full rate."""
        return [
            {"params": list(self.encoder.parameters()), "lr": lr * encoder_scale},
            {"params": list(self.head.parameters()) + [self.log_std]
             + list(self.critic.parameters()), "lr": lr},
        ]


def _image(value) -> torch.Tensor:
    """`[H, W, 3]` uint8 (or float) -> `[3, H, W]` float in `[0, 1]`."""
    if isinstance(value, torch.Tensor):
        x = value.float()
    else:
        x = torch.as_tensor(np.ascontiguousarray(np.asarray(value)), dtype=torch.float32)
    if float(x.max()) > 1.0 + 1e-6:
        x = x / 255.0
    return x.permute(2, 0, 1).contiguous()


def alignment_input(rows: torch.Tensor, valid: torch.Tensor, mech: torch.Tensor,
                    action: torch.Tensor, in_hand: float) -> torch.Tensor:
    """`[PREDICTOR_IN]` for the frozen predictor `F`, from the policy's own rows.

    `F` was trained on the controller's slate -- three named references and the brick in the hand,
    in that order -- and the policy's rows are in the frame's own candidate order instead, so the
    entries do not line up object for object. The term is therefore a predictability constraint on
    the representation rather than a calibrated forecast, and it is reported as one.
    """
    n = N_SLATE
    read = torch.cat([rows[:n][:, sl] for sl in _READ], dim=1)
    v = valid[:n].reshape(n, 1)
    per_row = torch.cat([v, v, read * v], dim=1)
    return torch.cat([per_row.reshape(-1), mech[:MECH_W], action[:4],
                      torch.as_tensor([float(in_hand)])])


def alignment_target(rows_next: torch.Tensor, valid_next: torch.Tensor) -> torch.Tensor:
    """`[N_SLATE * ROW_W]`: the next frame's own rows, in the same read fields, detached."""
    n = N_SLATE
    read = torch.cat([rows_next[:n][:, sl] for sl in _READ], dim=1)
    return (read * valid_next[:n].reshape(n, 1)).reshape(-1)
