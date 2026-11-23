import unittest
import torch
from exec_comp.hat_codec import hat_rotation_quat,action_to_keypoints
from exec_comp.geometry import qnormalize,qapply,qmul,q_to_6d


def rows(q):
    basis=torch.eye(3).expand(len(q),-1,-1)
    matrix=qapply(q[:,None].expand(-1,3,-1),basis).transpose(1,2)
    return matrix[:,:2].flatten(1)


class HatCodecTests(unittest.TestCase):
    def test_reference_slerp_matches_scipy(self):
        import numpy as np
        from scipy.spatial.transform import Rotation,Slerp
        from exec_comp.hat_playback import quaternion_slerp
        torch.manual_seed(13);q=qnormalize(torch.randn(2,4))
        expected=Slerp([0,1],Rotation.from_quat(q[:,[1,2,3,0]].numpy()))([.25,.5,.75]).as_quat()
        actual=quaternion_slerp(q[:1].repeat(3,1),q[1:].repeat(3,1),torch.tensor([[.25],[.5],[.75]]))
        dots=np.abs((actual.numpy()*expected[:,[3,0,1,2]]).sum(-1))
        self.assertLess(float(np.abs(1-dots).max()),1e-6)

    def test_hat_rows_are_converted_to_cerebellum_xz_columns(self):
        torch.manual_seed(5);q=qnormalize(torch.randn(100,4))
        recovered=hat_rotation_quat(rows(q))
        self.assertLess(float((1-(q*recovered).sum(-1).abs()).abs().max()),1e-6)

    def test_online_inspire_mount_is_inverted_for_both_hands(self):
        torch.manual_seed(9);q=qnormalize(torch.randn(10,4));p=torch.randn(10,3)
        action=torch.zeros(10,128)
        identity=torch.tensor([1.,0,0,0,1.,0])
        for start in (89,0,80,30):action[:,start+3:start+9]=identity
        for side,start in ((0,80),(1,30)):
            mount=q.new_tensor([.70710678,0,0,.70710678] if side==0 else [0,.70710678,-.70710678,0]).expand_as(q)
            action[:,start:start+3]=p+qapply(q,q.new_tensor([.0415,0,0]).expand_as(p))
            action[:,start+3:start+9]=rows(qnormalize(qmul(q,mount)))
        result=action_to_keypoints(action)
        for index in (2,3):
            torch.testing.assert_close(result[:,index,:3],p,atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(result[:,index,3:],q_to_6d(q),atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main()
