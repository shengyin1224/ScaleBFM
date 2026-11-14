"""Build small independent-hand focus sidecars; original August body NPZs stay immutable."""
import argparse
import ast
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import yaml

def load_brain_detector(path):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'transition_labels')
    source = ast.unparse(node)
    scope = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['transition_labels'], hashlib.sha256(source.encode()).hexdigest()

def next_edge(mask, fps):
    edges = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    following = np.searchsorted(edges, np.arange(len(mask)), side='right')
    ends = np.r_[edges, len(mask)][following]
    seconds = np.minimum((ends - np.arange(len(mask))) / fps, 2.)
    return np.where(mask, -seconds, seconds).astype(np.float32)

def nearest_indices(n, source_fps, target_fps=50.):
    duration = (n - 1) / source_fps
    times = np.arange(0., duration, 1. / target_fps, dtype=np.float32)
    return np.clip(np.rint(times / duration * (n - 1)).astype(np.int64), 0, n - 1)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', required=True)
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--brain-source', required=True)
    p.add_argument('--output-root', required=True)
    args = p.parse_args()
    source, output = Path(args.source_root), Path(args.output_root)
    labels = output / 'labels'
    labels.mkdir(parents=True, exist_ok=True)
    detector, detector_hash = load_brain_detector(args.brain_source)
    manifest = yaml.safe_load((source / 'manifests/unifolm_hat4_all.yaml').read_text())
    audit = dict(definition='exact human-policy transition_labels, independent left/right',
                 hand_order=['left','right'], brain_source=args.brain_source,
                 detector_ast_sha256=detector_hash, source_root=str(source),
                 body_data='original August v2 NPZs, not copied or modified',
                 episodes=0, frames=0, focus_frames_lr=[0,0], events_lr=[0,0],
                 source_files={})
    for i, (name, body_path) in enumerate(manifest.items()):
        family, ep = name.rsplit('_ep_', 1)
        folder = 'finger_keypoints_rx90_ry180' if 'Brainco' in family else 'finger_keypoints_reverse_thumb_swap'
        finger_file = Path(args.dataset_root) / family / folder / f'ep_{ep}.pkl'
        raw_bytes = finger_file.read_bytes()
        raw = pickle.loads(raw_bytes)
        fps = float(raw['fps'])
        left, right = np.asarray(raw['finger_pos_left']), np.asarray(raw['finger_pos_right'])
        if left.shape != right.shape or left.shape[1:] != (5,3) or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f'Invalid fingertip geometry: {finger_file}')
        n = len(left)
        states = np.zeros((n,128), np.float32)
        # Same offsets and same five-tip order as human-policy's 128-D state.
        states[:,13:28] = left.reshape(n,15)
        states[:,43:58] = right.reshape(n,15)
        _, _, weights, events = detector(states, fps)
        mask30 = weights > 1
        edge30 = np.stack([next_edge(mask30[:,h], fps) for h in range(2)], axis=1)
        nearest = nearest_indices(n, fps)
        with np.load(body_path) as body:
            if len(nearest) != len(body['joint_pos']):
                raise ValueError(f'Frame alignment mismatch: {name}')
        mask, edge = mask30[nearest], edge30[nearest]
        valid = np.ones_like(mask, dtype=bool)
        np.savez_compressed(labels / f'{name}.npz', focus_mask_lr=mask,
                            focus_time_to_next_edge_s_lr=edge, focus_valid_lr=valid)
        audit['episodes'] += 1
        audit['frames'] += len(mask)
        audit['focus_frames_lr'] = (np.asarray(audit['focus_frames_lr']) + mask.sum(0)).tolist()
        for hand, *_ in events:
            audit['events_lr'][hand] += 1
        audit['source_files'][name] = dict(finger_file=str(finger_file), finger_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                                           source_frames=n, target_frames=len(mask))
        if (i+1) % 100 == 0:
            print(f'Built {i+1}/{len(manifest)} independent focus labels', flush=True)
    for split in ('train','heldout','all'):
        content = (source / f'manifests/unifolm_hat4_{split}.yaml').read_bytes()
        dest = output / 'manifests' / f'unifolm_hat4_{split}.yaml'
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(content)
    audit['focus_fraction_lr'] = (np.asarray(audit['focus_frames_lr']) / audit['frames']).tolist()
    (output / 'audit.json').write_text(json.dumps(audit, indent=2))
    (output / 'READY').write_text(detector_hash + '\n')
    print(json.dumps({k:v for k,v in audit.items() if k != 'source_files'}, indent=2))

if __name__ == '__main__':
    main()
