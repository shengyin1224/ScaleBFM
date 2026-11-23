import xml.etree.ElementTree as ET
import torch
from exec_comp.common import ROOT, metadata, ACTIVE, HEAD_OFFSET
from exec_comp.geometry import qapply,qmul,pose9


class FK(torch.nn.Module):
    def __init__(self,xml=None):
        super().__init__()
        tree=ET.parse(xml or ROOT/"assets/g1_29dof.xml")
        root=tree.getroot().find("worldbody").find("body")
        names=[];parents=[];translations=[];rotations=[];axes=[];joint_ids=[]
        actor_names=metadata()["joint_names"]
        def visit(node,parent):
            index=len(names);names.append(node.attrib["name"]);parents.append(parent)
            translations.append([float(v) for v in node.attrib.get("pos","0 0 0").split()])
            rotations.append([float(v) for v in node.attrib.get("quat","1 0 0 0").split()])
            joints=node.findall("joint")
            if parent>=0:
                if len(joints)!=1:raise ValueError(f"Expected one hinge: {names[-1]}")
                joint_ids.append(actor_names.index(joints[0].attrib["name"]))
                axes.append([float(v) for v in joints[0].attrib.get("axis","0 0 1").split()])
            else:joint_ids.append(0);axes.append([0.,0.,0.])
            for child in node.findall("body"):visit(child,index)
        visit(root,-1)
        self.names=names;self.parents=parents;self.joint_ids=joint_ids
        self.selected=[names.index(n) for n in metadata()["selected_body_names"]]
        self.register_buffer('selected_index',torch.tensor(self.selected,dtype=torch.long))
        self.register_buffer('active_index',torch.tensor(ACTIVE,dtype=torch.long))
        self.register_buffer("translations",torch.tensor(translations))
        self.register_buffer("rotations",torch.tensor(rotations))
        self.register_buffer("axes",torch.tensor(axes))
        self.register_buffer("head_offset",torch.tensor(HEAD_OFFSET))

    def forward(self,q):
        positions=[torch.zeros_like(q[...,:3])]
        identity=torch.zeros_like(q[...,:4]);identity[...,0]=1
        orientations=[identity]
        for i in range(1,len(self.names)):
            parent=self.parents[i]
            positions.append(positions[parent]+qapply(orientations[parent],self.translations[i].expand_as(positions[parent])))
            angle=q[...,self.joint_ids[i]:self.joint_ids[i]+1]*.5
            joint=torch.cat((angle.cos(),self.axes[i]*angle.sin()),-1)
            orientations.append(qmul(orientations[parent],qmul(self.rotations[i].expand_as(joint),joint)))
        return torch.stack(positions,-2),torch.stack(orientations,-2)

    def keypoints(self,q):
        p,r=self(q);p=p.index_select(-2,self.selected_index).index_select(-2,self.active_index);r=r.index_select(-2,self.selected_index).index_select(-2,self.active_index)
        p=torch.stack((p[...,0,:],p[...,1,:]+qapply(r[...,1,:],self.head_offset.expand_as(p[...,1,:])),p[...,2,:],p[...,3,:]),-2)
        return pose9(p,r)
