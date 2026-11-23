"""One TorchScript interface for A1/B1, optionally captured in a CUDA Graph."""
import torch
from exec_comp.geometry import six_to_q,qmul,qconj,qnormalize


class ScriptPredictor:
    def __init__(self,path,device,method,graph=False):
        if method not in ('a1','b1'):raise ValueError(method)
        self.method=method;self.checkpoint=str(path);self.device=device
        self.model=torch.jit.load(str(path),map_location=device).eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.use_graph=bool(graph);self.graph=None;self.static=None

    def unpack(self,history):
        n=history['q'].shape[1]
        values={k:history[k][0] for k in ('q','dq','kp')}
        values['root_pos']=history['root_pos'][0,None].expand(n,-1)
        values['root_quat']=history['root_quat'][0,None].expand(n,-1)
        if self.method=='b1':values['s']=history['s'][0]
        return values

    def compute(self,state,window):
        corrected=self.model.correct_kp(state,window)
        position=(corrected[...,:3]-window[...,:3])[None]
        if self.method=='b1':
            rotation=position.new_zeros(*position.shape[:-1],4);rotation[...,0]=1.
        else:rotation=qnormalize(qmul(six_to_q(corrected[...,3:]),qconj(six_to_q(window[...,3:]))))[None]
        return position,rotation

    @torch.no_grad()
    def predict_coefficients(self,history,window):
        if self.method!='b1' or not hasattr(self.model,'correct_coefficients'):
            raise ValueError('Native coefficient interface absent: select the coefficient-capable B1 export')
        if self.use_graph:raise ValueError('Coefficient-transition graph path has not been validated; use cuda_graph=false')
        return self.model.correct_coefficients(self.unpack(history),window[0])[None]

    @torch.no_grad()
    def __call__(self,history,window):
        if len(window)!=1:raise ValueError('Deployment predictor is batch1')
        state=self.unpack(history)
        if not self.use_graph:return self.compute(state,window[0])
        if self.graph is None:
            if not str(self.device).startswith('cuda'):raise ValueError('CUDA Graph needs CUDA')
            self.static={k:v.clone() for k,v in state.items()};self.window=window[0].clone()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):self.compute(self.static,self.window)
            torch.cuda.current_stream().wait_stream(stream)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self.output=self.compute(self.static,self.window)
        for key,value in state.items():self.static[key].copy_(value)
        self.window.copy_(window[0]);self.graph.replay()
        return self.output
