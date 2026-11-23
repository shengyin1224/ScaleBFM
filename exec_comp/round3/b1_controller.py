"""Simulation-side wrapper that makes G drop into the B controller's slot.

Same ``correct(history, goal) -> corrected chunk`` signature the round-2
execution controller exposes, so the collection loop and the closed-loop
evaluation can use G without a second integration path. One forward, no
optimization: that is the whole point of amortizing B.
"""
import torch
from exec_comp.common import HISTORY
from exec_comp.geometry import from_anchor,heading_anchor,to_anchor
from exec_comp.geometry import qapply
from exec_comp.models.amortized import AmortizedCorrection,encode_state


class B1Controller:
    def __init__(self,checkpoint,device="cuda:0"):
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        if saved.get("kind")!="amortized":raise ValueError(f"{checkpoint} is not a B1 network")
        self.model=AmortizedCorrection(saved["parameterization"]).to(device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval().requires_grad_(False)
        self.device=device;self.metadata=saved
        self.parameterization=saved["parameterization"]

    @torch.no_grad()
    def correct(self,histories,goal):
        """``histories`` is the round-2 History object or its dict of buffers."""
        data=histories.data if hasattr(histories,"data") else histories
        origin,heading=heading_anchor(data["root_p"][:,-1],data["root_q"][:,-1])
        state=data["s"][:,-HISTORY:]
        encoded=encode_state(to_anchor(data["kp"][:,-HISTORY:],origin,heading),
                             state[...,:29],state[...,29:58],
                             state[...,58:61],state[...,61:64])
        corrected=self.model(encoded,to_anchor(goal,origin,heading))
        return from_anchor(corrected,origin,heading)


class B1Predictor:
    """G plugged into the SAME A1Executor limits, LP, freeze and event stage."""
    def __init__(self,checkpoint,device='cuda:0'):
        self.controller=B1Controller(checkpoint,device);self.model=self.controller.model
        self.checkpoint=str(checkpoint);self.metadata=self.controller.metadata

    @torch.no_grad()
    def __call__(self,history,window):
        if 's' not in history:raise ValueError('B1 requires measured10-frame gyro/gravity history')
        state=history['s'][:,-HISTORY:]
        origin,heading=heading_anchor(history['root_pos'],history['root_quat'])
        encoded=encode_state(to_anchor(history['kp'],origin,heading),history['q'],history['dq'],
                             state[...,58:61],state[...,61:64])
        delta=self.model.residual(encoded,to_anchor(window,origin,heading))
        delta=qapply(heading[:,None,None].expand(*delta.shape[:-1],4),delta)
        rotation=delta.new_zeros(*delta.shape[:-1],4);rotation[...,0]=1.
        return delta,rotation
