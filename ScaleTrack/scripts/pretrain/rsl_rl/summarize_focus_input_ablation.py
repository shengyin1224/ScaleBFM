"""Summarize paired focus-input ablations on identical successful motions."""
import json
import math
from pathlib import Path
import torch

def summarize(directory):
    root = Path(directory)
    reports = json.loads((root / 'summary.json').read_text())
    values = {key: torch.load(root / f'{key}.per_motion.pt', map_location='cpu', weights_only=False) for key in reports}
    reference = next(iter(values.values()))
    common = torch.ones_like(reference['error_body_pos_g_max'], dtype=torch.bool)
    successes = {}
    for key, result in values.items():
        assert torch.equal(result['motion_ids'], reference['motion_ids'])
        assert torch.equal(result['focus_frame_count'], reference['focus_frame_count']), 'Ablating input must never alter statistical focus labels'
        successes[key] = result['error_body_pos_g_max'] <= .5
        common &= successes[key]
    common_focus = common & (reference['focus_frame_count'] > 0)
    selected = [f'error_{part}_{frame}_{kind}_focus' for part in ('left_wrist','right_wrist','head_proxy')
                for frame in ('global','pelvis_local') for kind in ('pos','rot')]
    summary = dict(label_sha256=next(iter(reports.values()))['label_sha256'],
                   statistical_focus_identical=True, common_success_motions=int(common.sum()),
                   common_success_with_focus_motions=int(common_focus.sum()), conditions={}, deltas={})
    assert len({r['label_sha256'] for r in reports.values()}) == 1
    for key, result in values.items():
        metrics = {}
        for name in selected:
            scale = 100. if '_pos_' in name else 180. / math.pi
            metrics[name] = float(result[name][common_focus].mean()) * scale
        summary['conditions'][key] = dict(success_count=int(successes[key].sum()),
                                         success_rate=float(successes[key].float().mean()),
                                         common_success_focus_errors_cm_deg=metrics,
                                         original_success_subset_metrics=reports[key]['metrics'])
    for version in ('july','august'):
        correct = summary['conditions'][f'{version}_correct']['common_success_focus_errors_cm_deg']
        constant = summary['conditions'][f'{version}_constant_0_2']['common_success_focus_errors_cm_deg']
        summary['deltas'][version] = {key:dict(constant_minus_correct=constant[key]-correct[key],
                                                 percent_change=(constant[key]/correct[key]-1)*100)
                                      for key in selected}
    (root / 'comparison.json').write_text(json.dumps(summary,indent=2))
    print('PAIRED_FOCUS_COMPARISON', json.dumps(summary),flush=True)
    return summary

if __name__ == '__main__':
    import sys
    summarize(sys.argv[1])
