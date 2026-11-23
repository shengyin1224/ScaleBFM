"""Separate optional ScaleBridge environment class. No existing class is patched."""
from env.motion_tracking_hat4_online import HAT4OnlineMotionTrackingEnv
from exec_comp.deployment_hooks import ExecCompHooks
from exec_comp.deployment_adapter import DeploymentAdapter


class ExecCompOnlineEnv(ExecCompHooks,HAT4OnlineMotionTrackingEnv):
    def __init__(self,config,metadata_dict,device):
        self._exec_adapter=None;self._exec_delta=None
        super().__init__(config,metadata_dict,device)
        options=config.get('exec_comp',{})
        if bool(options.get('enable',False)):
            if abs(float(self.dt)-.02)>1e-6:raise ValueError('Validated execution compensation requires50Hz control')
            self._exec_adapter=DeploymentAdapter(options,controller_metadata=metadata_dict,device=device)
