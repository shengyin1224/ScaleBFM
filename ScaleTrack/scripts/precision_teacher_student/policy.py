"""Deterministic, explicitly masked inference shared by PPO checks and DAgger."""
import torch


def disable_phase_weights(policy):
    # Deployment retains [task(255), mode(14)]; phase is task columns 253:255.
    with torch.no_grad():
        policy.actor_task_embedder.task_projection.weight[:, 253:255].zero_()
        policy.critic_task_embedder.task_projection.weight[:, -2:].zero_()


def mean_action(policy, obs, teacher=False, position_noise=None):
    prop = obs["teacher_policy"] if teacher else obs["policy"]
    task = (obs["teacher_policy_task"] if teacher else obs["policy_task"]).clone()
    task[..., 253:255] = 0
    if position_noise is not None:
        # A common localization translation, shared by links and future tokens.
        task[..., :42] += position_noise[:, None, None, :].expand(-1, 6, 14, -1).flatten(2)
        task[..., 42:84] += position_noise[:, None, None, :].expand(-1, 6, 14, -1).flatten(2)
    mode = torch.ones_like(obs["mode"]) if teacher else obs["mode"]
    mask = torch.ones_like(obs["mode_mapping"]) if teacher else obs["mode_mapping"]
    if mask.ndim == 2:
        mask = mask[:, None]
    if mode.ndim == 2:
        mode = mode[:, None].expand(-1, task.shape[1], -1)
    tokens = policy.actor_task_embedder(torch.cat((task * mask, mode), dim=-1))
    return policy.actor(prop, obs["action"], tokens)
