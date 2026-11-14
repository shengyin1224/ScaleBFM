#!/usr/bin/env python3
"""Prove that +4 zero-column checkpoint expansion preserves iter-0 actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tensordict import TensorDict

from my_rsl_rl.modules.actor_critic_humanoid_transformer import (
    ActorCriticHumanoidTransformer,
)


def observations(
    batch: int,
    policy_task_width: int,
    critic_task_width: int,
    critic_prop_width: int,
    phase: torch.Tensor | None = None,
) -> TensorDict:
    generator = torch.Generator().manual_seed(20260727)
    policy = torch.randn(batch, 3, 64, generator=generator)
    policy_task = torch.randn(batch, 6, 253, generator=generator)
    action = torch.randn(batch, 3, 29, generator=generator)
    mode = torch.randn(batch, 14, generator=generator)
    mapping = torch.ones(batch, 253)
    if policy_task_width == 257:
        assert phase is not None and phase.shape == (batch, 6, 4)
        policy_task = torch.cat((policy_task, phase), dim=-1)
        mapping = torch.cat((mapping, torch.ones(batch, 4)), dim=-1)
    return TensorDict(
        {
            "policy": policy,
            "policy_task": policy_task,
            "critic": torch.randn(
                batch, 3, critic_prop_width, generator=generator
            ),
            "critic_task": torch.randn(
                batch, 7, critic_task_width, generator=generator
            ),
            "action": action,
            "mode": mode,
            "mode_mapping": mapping,
        },
        batch_size=[batch],
    )


def build(obs: TensorDict) -> ActorCriticHumanoidTransformer:
    return ActorCriticHumanoidTransformer(
        obs=obs,
        obs_groups={},
        num_actions=29,
        use_transformer_critic=True,
        embedding_dim=256,
        num_heads=4,
        ff_dim=256,
        num_layers=4,
        task_embedder_hidden_dims=[],
    ).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    critic_prop_width = int(state["critic.prop_projection.weight"].shape[1])
    batch = 4
    phase = torch.tensor(
        [[[1.0, -0.4, 0.0, 1.2]] * 6, [[0.0, 1.2, 1.0, -0.5]] * 6] * 2,
        dtype=torch.float32,
    )
    old_obs = observations(batch, 253, 337, critic_prop_width)
    new_obs = observations(batch, 257, 341, critic_prop_width, phase)
    old_model = build(old_obs)
    new_model = build(new_obs)
    old_model.load_state_dict(state)
    resumed = new_model.load_state_dict(state)
    with torch.inference_mode():
        old_action = old_model.act_inference(old_obs.clone())
        new_action = new_model.act_inference(new_obs.clone())
    difference = torch.abs(old_action - new_action)
    expanded = list(new_model.last_load_zero_expanded_keys)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "expanded_keys": expanded,
        "load_classified_as_resume": bool(resumed),
        "max_abs_action_difference": float(difference.max()),
        "mean_abs_action_difference": float(difference.mean()),
        "bitwise_equal": bool(torch.equal(old_action, new_action)),
    }
    if len(expanded) != 2 or resumed or not torch.equal(old_action, new_action):
        raise AssertionError(report)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
