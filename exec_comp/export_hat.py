"""Offline held-out inference only: uses existing HAT loader, no sockets/cameras/robot."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import h5py
import cv2
from exec_comp.common import ROOT,save_json,digest,log
from exec_comp.hat_codec import action_to_keypoints
from exec_comp.geometry import pose9


def inference_policy_class(service,repo):
    """Restore the complete saved trunk; avoid fetching redundant initialization weights."""
    class OfflinePolicy(service.OnlineHATPolicy):
        def _load_policy(self,checkpoint):
            from hdt.policy import ACTPolicy
            from detr.models import detr_vae
            original=detr_vae.FrozenPatchTargetEncoder
            class InferenceEncoder(original):
                def __init__(self,name='dinov2_vits14',target_resolution_hw=None):
                    torch.nn.Module.__init__(self)
                    self.name=name;self.target_resolution_hw=tuple(target_resolution_hw) if target_resolution_hw else None
                    self.tubelet_size=1;self.unused_target=name.startswith('dinov2')
                    if self.unused_target:
                        self.kind='dinov2';self.patch_size=14;self.embed_dim=original.DINOV2_EMBED_DIM[name]
                        return
                    entry,_,_,width=original.VJEPA2_MODELS[name]
                    revision=original.VJEPA2_HUB_REF.split(':',1)[1]
                    candidates=[Path('/home/aigc/.cache/torch/hub')/('facebookresearch_vjepa2_'+revision),Path('/root/.cache/torch/hub')/('facebookresearch_vjepa2_'+revision)]
                    cached=next((p for p in candidates if p.exists()),None)
                    if cached is None:raise FileNotFoundError(f'Pinned VJEPA source cache missing: {candidates}')
                    encoder,predictor=torch.hub.load(str(cached),entry,source='local',pretrained=False)
                    del predictor
                    self.body=encoder.eval().requires_grad_(False);self.kind='vjepa2';self.patch_size=16
                    self.tubelet_size=int(getattr(encoder,'tubelet_size',2));self.embed_dim=width

                def forward(self,x):
                    if self.unused_target:raise RuntimeError('Training-only DINO targets must not run in prior-only validation')
                    return super().forward(x)
            config=json.loads((checkpoint.parent/'training_config.json').read_text())
            detr_vae.FrozenPatchTargetEncoder=InferenceEncoder
            try:policy=ACTPolicy(config['config']['policy_config'].copy())
            finally:detr_vae.FrozenPatchTargetEncoder=original
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            policy.load_state_dict(state,strict=True)
            if not torch.equal(policy.model.state_keep_mask.cpu(),state['model.state_keep_mask']):raise AssertionError('Checkpoint input mask mismatch')
            log('hat_strict_reload',keys=len(state),masked_indices=torch.nonzero(state['model.state_keep_mask']==0).flatten().tolist())
            _,preprocess=service.make_visual_encoder('ACT',None)
            return policy.eval().requires_grad_(False).to(self.device),preprocess,False
    return OfflinePolicy


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--bundle',required=True)
    p.add_argument('--split',default=str(ROOT/'configs/hat_heldout_split.json'))
    p.add_argument('--fps',type=float,default=30.);p.add_argument('--output',default='runs/round2/hat_predictions')
    p.add_argument('--limit',type=int,default=25);p.add_argument('--max-queries',type=int,default=0)
    # Round 3 queries on the replan cadence measured from the deployment logs.
    # The default reproduces the round-2 3Hz schedule exactly.
    p.add_argument('--replan-interval-s',type=float,default=1/3)
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    repo=Path(args.repo);bundle=Path(args.bundle)
    for path in (repo/'hdt/online_hat_service.py',bundle/'policy_last.ckpt',bundle/'dataset_stats.pkl'):
        if not path.exists():raise FileNotFoundError(path)
    sys.path[:0]=[str(repo),str(repo/'hdt')]
    import hdt.online_hat_service as service
    from hdt.data_utils_hdt import _detect_valid_start_from_actions
    config=next(bundle.glob('act_input*.yaml'))
    training=json.loads((bundle/'training_config.json').read_text())
    training_split=Path(training['args']['dataset_json_path'])
    if not training_split.exists():raise FileNotFoundError(training_split)
    def heldout(d):return {(e['type'],name) for e in d['val'] for name in e['file_list']}
    if heldout(json.loads(training_split.read_text()))!=heldout(json.loads(Path(args.split).read_text())):raise ValueError('Held-out list differs from checkpoint training split')
    human_slowdown=float(training['args']['human_slow_down_factor'])
    nominal_fps=float(training['args']['robot_target_fps'])
    Policy=inference_policy_class(service,repo)
    engine=Policy(config_path=config,checkpoint=bundle/'policy_last.ckpt',stats_path=bundle/'dataset_stats.pkl',embodiment='dex5',device='cuda:0',wrist_conversion='inspire_hand_base')
    entries=json.loads(Path(args.split).read_text())['val'];episodes=[]
    for entry in entries:
        for filename in entry['file_list']:episodes.append((entry['type'],Path(entry['dataset_path'])/filename))
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=False);manifest=[];codec_error=0.
    for episode,(kind,path) in enumerate(episodes[:args.limit]):
        engine.stats=service._load_stats(bundle/'dataset_stats.pkl','human' if kind=='human' else 'dex5')
        if engine.stats['action_representation']!='original':raise ValueError('Audited checkpoint is original-action representation')
        with h5py.File(path) as f:
            source_fps=float(f.attrs['fps']);n=len(f['observation.state'])
            start=int(_detect_valid_start_from_actions(np.asarray(f['action'][:20])))
            human_factor=human_slowdown if kind=='human' else 1
            duration=(n-1-start)/source_fps*human_factor*nominal_fps/args.fps
            # Align reception to50Hz control ticks:17,17,16 steps per second at
            # the 3Hz default, which is the schedule the previous simulation
            # test used; --replan-interval-s selects a measured cadence instead.
            rate=1./float(args.replan_interval_s)
            times=np.ceil(np.arange(int(np.floor(duration*rate))+1,dtype=np.float64)*50/rate)/50
            if args.max_queries:times=times[:args.max_queries]
            frames=np.rint(start+times*source_fps/human_factor*args.fps/nominal_fps).astype(int).clip(start,n-1)
            image_keys=[k for k in f if k.startswith('observation.image.')]
            camera='observation.image.top' if 'observation.image.top' in f else ('observation.image.left' if 'observation.image.left' in f else image_keys[0] if len(image_keys)==1 else None)
            if camera is None:raise ValueError(f'Ambiguous camera mapping: {path}: {image_keys}')
            predictions=[]
            for query,frame in enumerate(frames):
                image=np.asarray(f[camera][int(frame)])
                if image.ndim==1:image=cv2.imdecode(image,cv2.IMREAD_COLOR)
                if image is None:raise ValueError(f'Image decoding failed: {path}:{frame}')
                image=cv2.resize(image,(320,240)).transpose(2,0,1)[None]
                raw=np.asarray(f['observation.state'][int(frame)],dtype=np.float32)
                # Exact formula in this checkpoint's older service snapshot,
                # whose normalization is inline in prepare_policy_state().
                normalized=((raw-engine.stats['qpos_mean'])/(engine.stats['qpos_std']+1e-6)).astype(np.float32)
                predictions.append(engine.infer(image,None,normalized_state=normalized))
            action=np.stack(predictions);canonical=action_to_keypoints(action).numpy()
            for index,start_index in enumerate((89,0,80,30)):
                p=action[...,start_index:start_index+3];q=service._rotation_6d_to_quat_wxyz(action[...,start_index+3:start_index+9])
                if index>=2:p,q=service._hand_base_to_g1_wrist(p,q,'left' if index==2 else 'right')
                expected=pose9(torch.tensor(p,dtype=torch.float32),torch.tensor(q,dtype=torch.float32)).numpy()
                difference=float(np.abs(expected-canonical[...,index,:]).max());codec_error=max(codec_error,difference)
                if difference>1e-5:raise AssertionError(f'Codec differs from online service: {difference}')
            initial=np.asarray(f['observation.state'][start],dtype=np.float32)
            destination=output/f'episode_{episode:02d}.npz'
            np.savez_compressed(destination,action=action,keypoints=canonical,query_time_s=times,source_frames=frames,
                                initial_state=initial,initial_keypoints=action_to_keypoints(initial).numpy(),
                                fingertip_local=action[...,np.r_[13:28,43:58]].reshape(len(times),100,2,5,3))
            row={'episode':episode,'source':str(path),'type':kind,'camera_key':camera,'source_fps':source_fps,'chunk_fps':args.fps,'queries':len(times),
                 'duration_s':duration,'source_start':start,'human_slowdown_factor':human_factor,'file':str(destination),'grasp_window':'N/A: no focus/grasp dataset fields'}
            manifest.append(row);log('hat_episode_exported',**row)
    save_json(output/'manifest.json',{'episodes':manifest,'checkpoint_sha256':digest(bundle/'policy_last.ckpt'),'stats_sha256':digest(bundle/'dataset_stats.pkl'),
        'chunk_fps':args.fps,'chunk_length':100,'replan_hz':1./float(args.replan_interval_s),
        'replan_interval_s':float(args.replan_interval_s),'inference':'prior only; no future actions or training targets passed to model',
        'coordinate_frame':'HAT initial-heading/opening-head frame; wrist mount converted to G1 wrist_yaw; canonical rotation uses x/z axes',
        'image_preprocessing':'Same online class and ACT transform; HDF JPEG decoded with training cv2 channel convention, no additional color swap',
        'retiming':'Human observation stream stretched4x like training action interpolation; robot source50Hz maps by nearest timestamp; 30/fps scales entire observation timeline to playback speed',
        'smoke':bool(args.max_queries or args.limit<25),'repo':str(repo),'training_split':str(training_split),'codec_vs_online_max_abs':codec_error,
        'source_sha256':{str(p.relative_to(repo)):digest(p) for p in (repo/'hdt/policy.py',repo/'hdt/detr/models/detr_vae.py',repo/'hdt/detr/models/backbone.py')}})


if __name__=='__main__':main()
