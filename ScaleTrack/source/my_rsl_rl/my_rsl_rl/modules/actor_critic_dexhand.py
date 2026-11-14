"""MLP + 1-D CNN actor-critic for the object-blind dexhand GMT policy."""

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal


def _activation(name: str) -> nn.Module:
    table = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return table[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"unsupported activation {name!r}") from exc


def _mlp(input_dim: int, hidden_dims: list[int] | tuple[int, ...], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), _activation(activation)))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class DexHandMotionEncoder(nn.Module):
    """TWIST's 20-step encoder with a shape-safe computed output width."""

    def __init__(self, token_dim: int, steps: int, latent_dim: int, activation: str):
        super().__init__()
        if steps != 20:
            raise ValueError(f"DexHandMotionEncoder expects 20 reference steps, got {steps}")
        channels = 20
        self.steps = steps
        self.token_dim = token_dim
        self.projection = nn.Sequential(nn.Linear(token_dim, 3 * channels), _activation(activation))
        self.convolution = nn.Sequential(
            nn.Conv1d(3 * channels, 2 * channels, kernel_size=6, stride=2),
            _activation(activation),
            nn.Conv1d(2 * channels, channels, kernel_size=4, stride=2),
            _activation(activation),
            nn.Flatten(),
        )
        with torch.no_grad():
            conv_width = self.convolution(torch.zeros(1, 3 * channels, steps)).shape[-1]
        self.output = nn.Linear(conv_width, latent_dim)

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        batch = motion.shape[0]
        projected = self.projection(motion.reshape(batch * self.steps, self.token_dim))
        encoded = self.convolution(projected.reshape(batch, self.steps, -1).transpose(1, 2))
        return self.output(encoded)


class DexHandPolicy(nn.Module):
    def __init__(
        self,
        motion_steps: int,
        motion_token_dim: int,
        proprio_dim: int,
        latent_dim: int,
        num_actions: int,
        hidden_dims: list[int],
        activation: str,
    ):
        super().__init__()
        self.motion_steps = motion_steps
        self.motion_token_dim = motion_token_dim
        self.motion_width = motion_steps * motion_token_dim
        self.motion_encoder = DexHandMotionEncoder(motion_token_dim, motion_steps, latent_dim, activation)
        self.backbone = _mlp(
            proprio_dim + motion_token_dim + latent_dim,
            hidden_dims,
            num_actions,
            activation,
        )

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        motion_flat = raw[:, : self.motion_width]
        proprio = raw[:, self.motion_width :]
        motion = motion_flat.reshape(-1, self.motion_steps, self.motion_token_dim)
        latent = self.motion_encoder(motion)
        return self.backbone(torch.cat((proprio, motion[:, 0], latent), dim=-1))


class ActorCriticDexHand(nn.Module):
    """Asymmetric actor-critic whose actor path cannot access object features."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_hidden_dims: list[int] = [512, 512, 256, 128],
        critic_hidden_dims: list[int] = [512, 512, 256, 128],
        motion_latent_dim: int = 128,
        activation: str = "silu",
        init_noise_std: float = 0.135335283,
        noise_std_type: str = "log",
        std_clamp_range: tuple[float, float] | list[float] = (0.001, 1.0),
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = False,
        **kwargs: dict[str, Any],
    ):
        super().__init__()
        if kwargs:
            print(f"ActorCriticDexHand ignored configuration keys: {sorted(kwargs)}")
        if noise_std_type != "log":
            raise ValueError("ActorCriticDexHand uses a clamped log_std parameter")
        task = obs["policy_task"]
        if task.ndim != 3:
            raise ValueError(f"policy_task must be (N, future_steps, token_dim), got {tuple(task.shape)}")
        motion_steps, token_dim = task.shape[1:]
        proprio_dim = obs["policy"].flatten(start_dim=1).shape[-1]
        actor_raw_dim = task.flatten(start_dim=1).shape[-1] + proprio_dim
        privileged_dim = obs["critic"].flatten(start_dim=1).shape[-1]

        self.obs_groups = obs_groups
        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization
        self.actor = DexHandPolicy(
            motion_steps,
            token_dim,
            proprio_dim,
            motion_latent_dim,
            num_actions,
            actor_hidden_dims,
            activation,
        )
        self.critic_motion_encoder = DexHandMotionEncoder(token_dim, motion_steps, motion_latent_dim, activation)
        self.critic = _mlp(
            proprio_dim + token_dim + motion_latent_dim + privileged_dim,
            critic_hidden_dims,
            1,
            activation,
        )
        self.motion_steps = motion_steps
        self.motion_token_dim = token_dim
        self.motion_width = motion_steps * token_dim
        self.register_buffer("actor_obs_mean", torch.zeros(actor_raw_dim))
        self.register_buffer("actor_obs_var", torch.ones(actor_raw_dim))
        self.register_buffer("actor_obs_count", torch.tensor(0.0))
        self.log_std = nn.Parameter(torch.full((num_actions,), float(torch.log(torch.tensor(init_noise_std)))))
        self.std_clamp_range = (float(std_clamp_range[0]), float(std_clamp_range[1]))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def get_actor_raw(self, obs: TensorDict) -> torch.Tensor:
        # Deliberately only these two keys.  critic and bc_target are excluded.
        return torch.cat(
            (
                obs["policy_task"].flatten(start_dim=1),
                obs["policy"].flatten(start_dim=1),
            ),
            dim=-1,
        )

    def normalize_actor_raw(self, raw: torch.Tensor) -> torch.Tensor:
        if not self.actor_obs_normalization:
            return raw
        return (raw - self.actor_obs_mean) / torch.sqrt(self.actor_obs_var + 1.0e-8)

    @torch.no_grad()
    def set_actor_normalization(
        self, mean: torch.Tensor, variance: torch.Tensor, count: int | float
    ) -> None:
        if mean.shape != self.actor_obs_mean.shape or variance.shape != self.actor_obs_var.shape:
            raise ValueError(
                f"normalizer shape mismatch: expected {tuple(self.actor_obs_mean.shape)}, "
                f"got mean={tuple(mean.shape)}, variance={tuple(variance.shape)}"
            )
        self.actor_obs_mean.copy_(mean)
        self.actor_obs_var.copy_(variance.clamp_min(1.0e-8))
        self.actor_obs_count.fill_(float(count))

    def actor_mean(self, obs: TensorDict) -> torch.Tensor:
        return self.actor(self.normalize_actor_raw(self.get_actor_raw(obs)))

    def _update_distribution(self, obs: TensorDict) -> None:
        mean = self.actor_mean(obs)
        log_min = float(torch.log(torch.tensor(self.std_clamp_range[0])))
        log_max = float(torch.log(torch.tensor(self.std_clamp_range[1])))
        std = torch.exp(torch.clamp(self.log_std, log_min, log_max)).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        return self.actor_mean(obs)

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        raw = self.normalize_actor_raw(self.get_actor_raw(obs))
        motion = raw[:, : self.motion_width].reshape(-1, self.motion_steps, self.motion_token_dim)
        proprio = raw[:, self.motion_width :]
        latent = self.critic_motion_encoder(motion)
        privileged = obs["critic"].flatten(start_dim=1)
        features = torch.cat((proprio, motion[:, 0], latent, privileged), dim=-1)
        return self.critic(features)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(-1)

    @property
    def actor_parameters(self):
        return [*self.actor.parameters(), self.log_std]

    @property
    def critic_parameters(self):
        return [*self.critic_motion_encoder.parameters(), *self.critic.parameters()]

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True

