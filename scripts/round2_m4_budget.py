"""L-only time-budget amendment; the registered training implementation is reused unchanged."""
import argparse
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.round2.common import setup, sha, write_json, utc
import round2_m4


def budget_record(root):
    receipt=json.loads((root/'provenance/round2_budget_receipt.json').read_text())
    if not receipt['remote_main_verified']:
        raise ValueError('Budget amendment was not verified on remote main')
    for name,value in receipt['files_sha256'].items():
        if sha(root/name)!=value:raise ValueError('Budget code differs from committed receipt')
    cfg=json.loads((root/'configs/round2_budget_amendment_v1.json').read_text())
    assert cfg['m4_S_single_run_seconds']==21600 and cfg['m4_total_seconds']==108000
    assert cfg['m4_L_single_run_seconds']==43200
    return cfg,receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--remaining-m4-seconds',type=float,required=True)
    args=p.parse_args();root=args.root.resolve();cfg,splits=setup(root)
    amendment,receipt=budget_record(root)
    if not 0<args.remaining_m4_seconds<=amendment['m4_total_seconds']:
        p.error('Invalid remaining aggregate budget')
    limit=min(amendment['m4_L_single_run_seconds'],args.remaining_m4_seconds)
    out=root/'results/round2-v1'/('head-L-seed'+str(args.seed))
    if out.exists():raise FileExistsError('Existing training evidence must not be overwritten')
    out.mkdir()
    write_json(out/'budget_amendment.json',{'started_utc':utc(),'receipt':receipt,
               'effective_single_run_seconds':limit,'remaining_total_seconds':args.remaining_m4_seconds,
               'original_training_implementation_unchanged':True})
    cfg['stop']['m4_single_run_seconds']=limit
    signal.signal(signal.SIGTERM,round2_m4.interrupted);signal.signal(signal.SIGINT,round2_m4.interrupted)
    try:
        round2_m4.train(root,cfg,splits,out,'L',args.seed)
    except BaseException as exc:
        write_json(out/'error.json',{'status':'failed','error_type':type(exc).__name__,
                   'phase':'train','updated_utc':utc(),'automatic_retry':False})
        raise


if __name__=='__main__':main()
