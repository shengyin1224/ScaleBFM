"""Independent FOCUS v4 adapter; append +focus_handoff=v4 to opt in.

The env choice still selects online/adaptive execution. This module never
patches an existing class, worker, module global, or configuration file.
"""

from contextlib import contextmanager
from importlib import import_module

from loguru import logger

from scalebridge.online.focus_handoff_v4 import upgrade_per_hand_handoff_v4


@contextmanager
def _legacy_worker_version(config):
    # Existing metadata setup and event worker accept v3. Its event detector
    # is identical to legacy; only the new controller owns the OR decision.
    original = config["focus_handoff_version"]
    config["focus_handoff_version"] = "v3"
    try:
        yield
    finally:
        config["focus_handoff_version"] = original


class FocusV4Mixin:
    def _setup_metadata(self):
        if self.cfg.get("focus_handoff_version") != "v4":
            raise ValueError("+focus_handoff=v4 requires focus_handoff_version=v4")
        if not self.cfg.get("focus_handoff_enabled", False):
            raise ValueError("FOCUS v4 requires focus_handoff_enabled=true")
        if self.cfg.get("focus_handoff_source", "keypoints") != "keypoints":
            raise ValueError("FOCUS v4 OR entry requires the keypoints source")
        with _legacy_worker_version(self.cfg):
            super()._setup_metadata()
        upgrade_per_hand_handoff_v4(self.focus_handoff, self.cfg)
        logger.info("[Online HAT] FOCUS v4 enabled: full v3 gate OR legacy "
                    "measured-entry gate at {} m; execution env unchanged.",
                    self.cfg.get("focus_v4_v1_position_tolerance_m", .025))

    def _setup_hand_ik(self):
        with _legacy_worker_version(self.cfg):
            return super()._setup_hand_ik()


def environment_class(name, *, execution_compensation=False):
    if name not in ("motion_tracking_hat4_online", "motion_tracking_hat4_adaptive"):
        raise ValueError(f"FOCUS v4 does not support env={name!r}")
    if execution_compensation:
        from scalebridge.env.motion_tracking_exec_comp import _project  # bootstrap sibling package
        from exec_comp.round4.deployment_env import environment_class as compensated_class
        base = compensated_class(name)
    else:
        module = import_module(f"scalebridge.env.{name}")
        base = getattr(module, {"motion_tracking_hat4_online": "HAT4OnlineMotionTrackingEnv",
                                "motion_tracking_hat4_adaptive": "HAT4AdaptiveMotionTrackingEnv"}[name])
    return type(f"FocusV4{base.__name__}", (FocusV4Mixin, base), {})


class HAT4FocusV4Env:
    def __new__(cls, config, metadata_dict, device, focus_v4_base_environment,
                execution_base_environment=None):
        if (execution_base_environment is not None
                and execution_base_environment != focus_v4_base_environment):
            raise ValueError("FOCUS v4 and execution compensation must select the same env")
        options = config.get("exec_comp", {}) or {}
        env_type = environment_class(focus_v4_base_environment,
                                     execution_compensation=bool(options))
        return env_type(config, metadata_dict, device)
