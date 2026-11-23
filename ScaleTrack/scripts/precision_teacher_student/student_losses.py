"""Training-only regularizers. The deployed policy interface is unchanged."""
import torch


def sequence_indices(indices, num_envs, continuation):
    """Time-major rollout indices; never regularize across resets or rollouts."""
    time = indices // num_envs
    previous = indices - time.clamp(max=1) * num_envs
    previous2 = indices - time.clamp(max=2) * num_envs
    valid = (time >= 2) & continuation[indices] & continuation[previous]
    return previous2, previous, valid


def physical_difference(action_difference, scale):
    # Convert to radians, then use one shared unit for numerical conditioning.
    return action_difference * scale / scale.square().mean().sqrt().clamp_min(1e-8)


def consistency_loss(action, perturbed_action, scale):
    return physical_difference(perturbed_action - action, scale).square().mean()


def temporal_loss(current, previous, previous2, valid, wrist_speed, scale, arm_ids):
    """Suppress zigzags; use a weaker static slew penalty, with soft speed gates."""
    delta = physical_difference(current - previous, scale)[:, arm_ids]
    second = physical_difference(current - 2 * previous + previous2, scale)[:, arm_ids]
    speed_gate = 1 / (1 + (wrist_speed / 0.15).square())
    static_gate = (1 - wrist_speed / 0.05).clamp(0, 1)
    per_sample = speed_gate * second.square().mean(-1) + 0.25 * static_gate * delta.square().mean(-1)
    return (per_sample * valid).sum() / valid.sum().clamp_min(1)


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    values = [g.detach().square().sum() for g in gradients if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def calibrated_weight(supervision_norm, regularizer_norm, fraction=0.1):
    if regularizer_norm <= 1e-12 or supervision_norm <= 1e-12:
        raise ValueError("Cannot calibrate from a zero gradient")
    return min(1000.0, max(1e-4, fraction * supervision_norm / regularizer_norm))
