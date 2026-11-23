"""Refresh evidence receipts without changing network/runtime manifest bindings."""
import json
from exec_comp.common import ROOT,digest,save_json
from exec_comp.round3.metrics import acceptance,verdict


def refresh():
    output={}
    for profile,label,flags in [('b1','b1_lowdim',{}),
            ('b1_coeffblend_candidate','b1_lowdim_coeffblend_v1',{'coefficient_transition':True}),
            ('b1_activation_candidate','b1_lowdim_activation_v1',{'activation_ramp':True})]:
        folder=ROOT/'releases/round4'/profile;manifest=folder/'manifest.json'
        if not manifest.exists():continue
        binding=json.loads(manifest.read_text());rows={}
        runtime=all(digest(ROOT/'round4'/name)==sha for name,sha in binding.get('runtime_source_sha256',{}).items())
        network=digest(folder/'b1_correct_kp.pt')==binding['artifact_sha256']
        for pop,subdir,suffix,raw in [('brain','brain_evaluation','','raw_stall-on'),
                ('motion','closed_loop','','raw_stall-on'),('dr','closed_loop','_dr','raw_dr')]:
            path=ROOT/'runs/round4'/subdir/(label+suffix)/'metrics.json'
            if not path.exists():rows[pop]={'status':'pending'};continue
            row=json.loads(path.read_text());opts=row['round4_execution']
            size=row['robot_episodes'] if pop=='brain' else row['metrics']['all']['clips']
            if row.get('truncated_smoke') or size!=(10 if pop=='brain' else 300):raise ValueError('Incomplete evaluation cannot count')
            for key in ('coefficient_transition','activation_ramp'):
                if bool(opts.get(key,False))!=flags.get(key,False):raise ValueError('Receipt points to a different runtime variant')
            baseline=json.loads((ROOT/'runs/round3'/subdir/raw/'metrics.json').read_text())
            values=acceptance(row,baseline);gates=verdict(values)
            fraction=row['executor']['round4']['fallback_fraction']
            gates['fallback_fraction']=fraction<=(.05 if pop=='dr' else .02)
            rows[pop]=dict(status='passed' if all(v is True for v in gates.values()) else 'failed',
                gates=gates,metrics=values,fallback_fraction=fraction,metrics_sha256=digest(path))
        result=dict(profile=profile,evaluations=rows,runtime_binding_matches=runtime,artifact_hash_matches=network,
            offline_all_gates_passed=runtime and network and all(v['status']=='passed' for v in rows.values()),
            ready_for_robot=False,latency_4080s='deferred until user notifies idle window',
            note='Evidence receipt only; does not modify or promote the bound model/runtime manifest',real_robot_run=False)
        save_json(folder/'evaluation_status.json',result);output[profile]=result
    return output


if __name__=='__main__':refresh()
