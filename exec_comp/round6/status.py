"""Fast read-only status from operational logs; no reports or plots written."""
import argparse,json,time
from pathlib import Path


def latest(path):
    if not path.exists():return {}
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-32768));lines=f.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:row=json.loads(line)
        except ValueError:continue
        if row.get('event') in ('r6_collect_progress','r6_collect_complete'):return row
    return {}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/dev/shm/exec_comp_round6/fullscale_v2');a=p.parse_args();root=Path(a.root)
    state=json.loads((root/'pipeline_state.json').read_text());jobs=json.loads((root/'collect_jobs.json').read_text());progress=[]
    for j in jobs:
        r=latest(root/f"collect_base_{j['split']}_{j['gpu']}.log");done=r.get('transitions',0);elapsed=r.get('elapsed_s',0);speed=done/elapsed if elapsed else 0
        progress.append(dict(gpu=j['gpu'],split=j['split'],logged_transitions=done,target=j['transitions'],estimated_remaining_s=(j['transitions']-done)/speed if speed else None,falls=r.get('falls')))
    print(json.dumps(dict(status=state['status'],phase=state['phase'],base_progress=progress,
        logged_transitions=sum(p['logged_transitions'] for p in progress),target=sum(p['target'] for p in progress),
        estimated_collection_remaining_minutes=max((p['estimated_remaining_s'] or 0) for p in progress)/60,
        final_real_check_consumed=state['final_direction_check_consumed'],updated_wall_s=state['updated_wall_s']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
