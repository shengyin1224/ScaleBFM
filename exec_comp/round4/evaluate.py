"""Reuse R3's exact physical harness; only replace the independent executor."""
import argparse
from pathlib import Path
import runpy
import sys
from exec_comp.common import ROOT,save_json,digest
from exec_comp.round4 import context


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--population',choices=('brain','motion'),required=True)
    parser.add_argument('--basis',default=None)
    parser.add_argument('--wrists-only',action='store_true')
    parser.add_argument('--fallback-off',action='store_true')
    parser.add_argument('--coefficient-model',action='store_true')
    parser.add_argument('--coefficient-transition',action='store_true')
    parser.add_argument('--activation-ramp',action='store_true')
    options,remaining=parser.parse_known_args()
    context.OPTIONS.update(vars(options))
    context.OPTIONS['fallback_off_simulation_only']=bool(options.fallback_off)
    if '--label' not in remaining:raise ValueError('Unique --label is required')
    label=remaining[remaining.index('--label')+1]
    folder='brain_evaluation' if options.population=='brain' else 'closed_loop'
    if '--output-root' in remaining:
        output_root=remaining[remaining.index('--output-root')+1]
    else:
        output_root=f'runs/round4/{folder}';remaining+=['--output-root',output_root]
    destination=ROOT/output_root/label
    import exec_comp.round3.a1_executor as module
    from exec_comp.round4.executor import Round4Executor
    loaded_source_hashes={name:digest(ROOT/'round4'/name) for name in ('executor.py','basis.py','coefficients.py')}
    original=module.A1Executor;module.A1Executor=Round4Executor
    # Kit may exit natively in app.close(), bypassing Python finally/atexit.
    # Commit audits at the metrics write, BEFORE lifecycle cleanup begins.
    import exec_comp.common as common
    original_save=common.save_json
    def save_before_cleanup(path,record):
        if Path(path)==destination/'metrics.json':
            for executor in context.INSTANCES:executor.audit.flush(destination)
            record['round4_execution']={**vars(options),'executor_sha256':loaded_source_hashes['executor.py'],
                'basis_sha256':loaded_source_hashes['basis.py'],'loaded_source_sha256':loaded_source_hashes,
                'source_hash_timing':'captured at module import before physics setup',
                'audit_trace_status':'saved before native Kit cleanup',
                'same_physical_harness':'round3, original vendor acceptance/RTC/hold'}
        return original_save(path,record)
    common.save_json=save_before_cleanup
    sys.argv=[sys.argv[0],*remaining]
    try:
        runpy.run_module('exec_comp.round3.eval_brain' if options.population=='brain' else 'exec_comp.round3.eval_switching',run_name='__main__')
    finally:
        module.A1Executor=original;common.save_json=original_save


if __name__=='__main__':main()
