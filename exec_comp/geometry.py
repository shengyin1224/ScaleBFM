import torch


def qconj(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def qmul(a, b):
    w = a[..., :1]*b[..., :1] - (a[..., 1:]*b[..., 1:]).sum(-1, keepdim=True)
    xyz = a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:] + torch.cross(a[..., 1:], b[..., 1:], dim=-1)
    return torch.cat((w, xyz), -1)


def qapply(q, v):
    xyz = q[..., 1:]
    t = 2*torch.cross(xyz, v, dim=-1)
    return v + q[..., :1]*t + torch.cross(xyz, t, dim=-1)


def qnormalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def q_to_6d(q):
    x = torch.zeros_like(q[..., 1:]); x[..., 0] = 1
    z = torch.zeros_like(x); z[..., 2] = 1
    return torch.cat((qapply(q, x), qapply(q, z)), -1)


def six_to_matrix(r):
    x = torch.nn.functional.normalize(r[..., :3], dim=-1)
    z = r[..., 3:] - (r[..., 3:]*x).sum(-1, keepdim=True)*x
    z = torch.nn.functional.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), -1)


def matrix_to_q(m):
    # Branch selection is piecewise differentiable and stable near identity.
    m00,m01,m02 = m[...,0,0],m[...,0,1],m[...,0,2]
    m10,m11,m12 = m[...,1,0],m[...,1,1],m[...,1,2]
    m20,m21,m22 = m[...,2,0],m[...,2,1],m[...,2,2]
    roots = torch.stack((1+m00+m11+m22,1+m00-m11-m22,1-m00+m11-m22,1-m00-m11+m22),-1).clamp_min(1e-8).sqrt()
    candidates = torch.stack((
        torch.stack((roots[...,0]**2,m21-m12,m02-m20,m10-m01),-1),
        torch.stack((m21-m12,roots[...,1]**2,m10+m01,m02+m20),-1),
        torch.stack((m02-m20,m10+m01,roots[...,2]**2,m12+m21),-1),
        torch.stack((m10-m01,m20+m02,m21+m12,roots[...,3]**2),-1)), -2)
    candidates = candidates/(2*roots[...,None].clamp_min(.1))
    index = roots.argmax(-1)[...,None,None].expand(*roots.shape[:-1],1,4)
    return qnormalize(candidates.gather(-2,index).squeeze(-2))


def six_to_q(r):
    return matrix_to_q(six_to_matrix(r))


def pose9(pos, quat):
    return torch.cat((pos,q_to_6d(quat)), -1)


def pose_normalize(p):
    matrix = six_to_matrix(p[...,3:])
    return torch.cat((p[...,:3],matrix[...,0],matrix[...,2]), -1)


def to_anchor(pose, origin, quat):
    while origin.ndim < pose.ndim:
        origin = origin.unsqueeze(-2); quat = quat.unsqueeze(-2)
    inverse = qconj(quat)
    p = qapply(inverse,pose[...,:3]-origin)
    x = qapply(inverse,pose[...,3:6]); z = qapply(inverse,pose[...,6:9])
    return torch.cat((p,x,z),-1)


def from_anchor(pose, origin, quat):
    while origin.ndim < pose.ndim:
        origin = origin.unsqueeze(-2); quat = quat.unsqueeze(-2)
    return torch.cat((qapply(quat,pose[...,:3])+origin,
                      qapply(quat,pose[...,3:6]),qapply(quat,pose[...,6:9])),-1)


def heading_anchor(root_pos,root_quat):
    # Remove arbitrary world xy/yaw, but retain height and gravity tilt: those
    # change contact dynamics and must not disappear from the learned state.
    w,x,y,z=root_quat.unbind(-1)
    yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    zero=torch.zeros_like(yaw)
    quat=torch.stack(((yaw*.5).cos(),zero,zero,(yaw*.5).sin()),-1)
    origin=torch.cat((root_pos[...,:2],torch.zeros_like(root_pos[...,2:3])),-1)
    return origin,quat


def integrate_quat(q, omega, dt=.02):
    angle = omega.norm(dim=-1,keepdim=True)*dt
    vector = omega*dt*.5*torch.sinc(angle/(2*torch.pi))
    return qnormalize(qmul(q,torch.cat((torch.cos(angle*.5),vector),-1)))


def integrate_root(root_p, root_q, state, next_state, dt=.02,com_offset=None):
    """Integrate root-link pose from Isaac's COM body-frame velocities.

    G1 pelvis COM offset is audited against the live simulation separately.
    Transform each endpoint velocity before trapezoidal integration.
    """
    offset = (state.new_tensor([0., 0., -.07605]) if com_offset is None else com_offset).expand_as(state[:, 65:68])
    omega, next_omega = state[:, 58:61], next_state[:, 58:61]
    quat = integrate_quat(root_q, .5*(omega+next_omega), dt)
    vel = state[:, 65:68]-torch.linalg.cross(omega, offset, dim=-1)
    next_vel = next_state[:, 65:68]-torch.linalg.cross(next_omega, offset, dim=-1)
    pos = root_p+.5*(qapply(root_q, vel)+qapply(quat, next_vel))*dt
    return torch.cat((pos[:, :2], next_state[:, 64:65]), -1), quat
