"""Optional environment class for the robot. No deployed class is patched."""
from scalebridge.env.motion_tracking_hat4_adaptive import HAT4AdaptiveMotionTrackingEnv
from exec_comp.round3.deployment_hooks import ExecCompRound3Hooks,assert_mutually_exclusive
from exec_comp.round3.deployment_adapter import Round3Adapter


class ExecCompRound3Env(ExecCompRound3Hooks,HAT4AdaptiveMotionTrackingEnv):
    def __init__(self,config,metadata_dict,device):
        self._exec_adapter=None;self._exec_sequence=None;self._exec_epoch=None
        options,compensation=assert_mutually_exclusive(config)
        super().__init__(config,metadata_dict,device)
        if bool(options.get("enable",False)) or bool(options.get("log_baseline",False)):
            if abs(float(self.dt)-.02)>1e-6:
                raise ValueError("The validated execution compensation requires 50 Hz control")
            self._exec_adapter=Round3Adapter(options,controller_metadata=metadata_dict,
                                             device=device,wrist_compensation=compensation)
