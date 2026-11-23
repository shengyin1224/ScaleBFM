"""B1: the amortized correction network G.

G replaces B's per-step optimization with a single forward. It sees only what
the robot can measure -- joint positions and velocities, base angular velocity
and gravity direction, the keypoints forward kinematics gives from those joint
angles, and the commanded K* window -- so nothing privileged leaks into
deployment. Privileged root state is used only to roll the world model forward
during training.

Two output parameterizations, as the task asks:

* ``lowdim`` (main): per keypoint and axis a constant offset and a ramp, 24
  numbers in total, expanded over the window. This is structurally smooth
  WITHIN one window. Parameters can still change between control calls, so
  closed-loop smoothness also needs the cross-step loss and runtime guards.
* ``per_frame`` (control): a residual for every frame, kept smooth only by the
  loss. It exists to show what the low-dimensional form costs.

Fingertips never reach G and are passed through by the caller unchanged.
"""
import torch
from exec_comp.common import CHUNK,HISTORY
from exec_comp.models.base import mlp

KEYPOINTS=4
AXES=3
# q, dq, angular velocity, gravity direction, plus FK keypoints per frame.
FRAME_DIM=29+29+3+3+KEYPOINTS*9
STATE_DIM=HISTORY*FRAME_DIM
PARAMETERIZATIONS=("lowdim","per_frame")


class AmortizedCorrection(torch.nn.Module):
    def __init__(self,parameterization="lowdim",chunk=CHUNK,width=512,
                 residual_scale_m=.05):
        super().__init__()
        if parameterization not in PARAMETERIZATIONS:
            raise ValueError(f"parameterization must be one of {PARAMETERIZATIONS}")
        self.parameterization=parameterization;self.chunk=int(chunk)
        self.residual_scale_m=float(residual_scale_m)
        outputs=(KEYPOINTS*AXES*2 if parameterization=="lowdim"
                 else self.chunk*KEYPOINTS*AXES)
        self.net=mlp([STATE_DIM+self.chunk*KEYPOINTS*9,width,width,width//2,outputs])
        self.register_buffer("input_mean",torch.zeros(STATE_DIM+self.chunk*KEYPOINTS*9))
        self.register_buffer("input_std",torch.ones(STATE_DIM+self.chunk*KEYPOINTS*9))
        self.register_buffer("phase",torch.linspace(0.,1.,self.chunk)[None,:,None,None])
        # A zero-initialized head starts G at "change nothing", so the first
        # training steps cannot throw the frozen cerebellum off its manifold.
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)

    def initialize_normalization(self,state,window):
        values=torch.cat((state,window.flatten(1)),1)
        with torch.no_grad():
            self.input_mean.copy_(values.mean(0))
            self.input_std.copy_(values.std(0).clamp_min(.02))

    def parameters_for(self,state,window):
        """Raw head output, before it becomes a per-frame residual."""
        values=torch.cat((state,window.flatten(1)),1)
        return self.net((values-self.input_mean)/self.input_std)

    def residual(self,state,window):
        """Position residual [B,chunk,4,3] in metres, bounded by construction."""
        raw=self.parameters_for(state,window)
        if self.parameterization=="lowdim":
            raw=raw.view(-1,2,KEYPOINTS,AXES)
            offset=torch.tanh(raw[:,0])[:,None]
            slope=torch.tanh(raw[:,1])[:,None]
            delta=offset+slope*self.phase
        else:
            delta=torch.tanh(raw.view(-1,self.chunk,KEYPOINTS,AXES))
        delta=delta*self.residual_scale_m
        # The same spherical bound the deployed stage enforces, applied here so
        # training never learns to rely on a correction it cannot send.
        norm=delta.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        return delta*torch.minimum(torch.ones_like(norm),self.residual_scale_m/norm)

    def forward(self,state,window):
        """Corrected command window; rotations and every other channel pass through."""
        delta=self.residual(state,window)
        return torch.cat((window[...,:3]+delta,window[...,3:]),-1)


def encode_state(keypoints,q,dq,angular_velocity,gravity):
    """Flatten the measurable history exactly as G was trained to read it.

    ``keypoints`` are anchored FK keypoints [B,HISTORY,4,9]; the rest are
    [B,HISTORY,...] measured channels. Nothing privileged appears here.
    """
    parts=(keypoints.flatten(2),q,dq,angular_velocity,gravity)
    return torch.cat(parts,-1).flatten(1)
