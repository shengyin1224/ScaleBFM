"""Tensor operations shared by variable-speed reference playback and its tests."""
import math
import torch


def eased_progress(step, start_step, end_step):
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step")
    u = min(1.0, max(0.0, (step - start_step) / (end_step - start_step)))
    return 1.0 - (1.0 - u) ** 2


def truncated_rates(shape, std, low=0.25, high=1.25):
    """Inverse-CDF sampling: truncation, not clipping a normal distribution."""
    if std <= 0 or not 0 < low < 1.0 < high:
        raise ValueError("Require std > 0 and 0 < low < 1 < high")
    a = 0.5 * (1 + math.erf((low - 1) / (std * math.sqrt(2))))
    b = 0.5 * (1 + math.erf((high - 1) / (std * math.sqrt(2))))
    p = a + (b - a) * torch.rand(shape)
    z = torch.erfinv((2 * p - 1).clamp(-1 + 1e-7, 1 - 1e-7))
    return (1 + std * math.sqrt(2) * z).clamp(low, high)


def interpolate_rates(curves, sim_time, interval):
    if interval <= 0:
        raise ValueError("playback_node_interval must be positive")
    shape = sim_time.shape
    t = sim_time.reshape(curves.shape[0], -1).clamp_min(0) / interval
    i = t.floor().long().clamp_max(curves.shape[1] - 1)
    j = (i + 1).clamp_max(curves.shape[1] - 1)
    w = (t - i).clamp(0, 1)
    return torch.lerp(curves.gather(1, i), curves.gather(1, j), w).reshape(shape)


def interpolate_samples(values, lower, upper, fraction, quaternion=False):
    a, b = values[lower], values[upper]
    w = fraction.reshape(*fraction.shape, *([1] * (a.ndim - fraction.ndim)))
    if not quaternion:
        return torch.lerp(a, b, w)
    # Shortest-arc SLERP, including q / -q and nearly identical rotations.
    dot = (a * b).sum(-1, keepdim=True)
    b = torch.where(dot < 0, -b, b)
    dot = dot.abs().clamp(0, 1)
    theta = torch.acos(dot.clamp_max(1 - 1e-7))
    denom = torch.sin(theta).clamp_min(1e-7)
    spherical = (torch.sin((1 - w) * theta) * a + torch.sin(w * theta) * b) / denom
    linear = torch.lerp(a, b, w)
    q = torch.where(dot > 0.9995, linear, spherical)
    return torch.nn.functional.normalize(q, dim=-1)


def precision_scales(reference_wrist_vel, sigma_pos_min, speed_range=(0.05, 0.10),
                     sigma_pos_max=0.10, sigma_rot_range=(math.radians(5), math.radians(20))):
    if speed_range[1] <= speed_range[0] or not 0 < sigma_pos_min <= sigma_pos_max:
        raise ValueError("Invalid precision scales or speed_range")
    speed = torch.linalg.vector_norm(reference_wrist_vel, dim=-1).amax(-1)
    alpha = ((speed - speed_range[0]) / (speed_range[1] - speed_range[0])).clamp(0, 1)
    return (sigma_pos_min + (sigma_pos_max - sigma_pos_min) * alpha,
            sigma_rot_range[0] + (sigma_rot_range[1] - sigma_rot_range[0]) * alpha)
