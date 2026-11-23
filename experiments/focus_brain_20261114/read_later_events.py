"""Read closed TensorBoard files without accumulating millions of training scalars."""
import json
from pathlib import Path
import struct
from tensorboard.compat.proto.event_pb2 import Event

TRACK = Path('/root/shengyin/ScaleBFM/ScaleTrack')
RUN = Path('/root/shengyin/ScaleBFM_focus_brain_20261114/experiments/focus_brain_20261114')
out = {}
for group in ['g1_hat4_focus_finetune', 'hat4_direct_ablation', 'precision_teacher_student']:
    for folder in (TRACK / 'logs/rsl_rl' / group).iterdir():
        if not folder.is_dir() or not any(x in folder.name for x in ['formal_core5_20260728b_F3h', 'formal_focusv2', 'F3hw4', 'official_threeway', 'direct_20261013_', 'precision_20261013_']):
            continue
        tags = {}
        for path in folder.glob('events*'):
            with path.open('rb') as f:
                while header := f.read(12):
                    if len(header) != 12:
                        break
                    n = struct.unpack('<Q', header[:8])[0]
                    record = f.read(n)
                    if len(f.read(4)) != 4:
                        break
                    if b'Eval_Test/' not in record:
                        continue
                    event = Event.FromString(record)
                    for value in event.summary.value:
                        if value.tag.startswith('Eval_Test/'):
                            tags[value.tag] = [[event.step, value.simple_value]]
        out[folder.name] = tags
        print(folder.name, {k: v[-1] for k,v in tags.items() if 'success_rate' in k}, flush=True)
        (RUN / 'later_audit/training_validation_last.json').write_text(json.dumps(out, indent=2))
