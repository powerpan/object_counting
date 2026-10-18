"""Rebuild fold decisions and scalar metrics without the selection helper."""
import argparse
import json
import math
from pathlib import Path
import statistics
from round3_common import sources,read,sha,write_json
from audit_b0_results import read_rows


def metrics(rows):
    e=[r['prediction']-r['truth'] for r in rows]
    return {'n':len(rows),'mae':statistics.mean(map(abs,e)),
            'rmse':math.sqrt(statistics.mean(v*v for v in e)),
            'game1':statistics.mean(r['game1'] for r in rows),'game2':statistics.mean(r['game2'] for r in rows)}


def choose(names,variants,base,order):
    m={v:metrics([variants[v][n] for n in names]) for v in order}
    b=metrics([base[n] for n in names]);eligible=[v for v in order if m[v]['rmse']<=b['rmse']]
    key=('mae','rmse') if eligible else ('rmse','mae')
    return sorted(eligible or order,key=lambda v:(m[v][key[0]],m[v][key[1]],order.index(v)))[0]


def main(root):
    cfg,splits,_=sources(root);out=root/'results/round3-v1/m3-crossfit';s=read(out/'summary.json')
    if (out/'audit.json').exists():raise FileExistsError('Existing crossfit audit')
    base_rows=read_rows(root/cfg['baseline_result']/'predictions.jsonl');names=[r['image'] for r in base_rows]
    base={r['image']:r for r in base_rows};variants={}
    for v in cfg['m3']['runs']:
        d=root/'results/round3-v1'/('full-'+v);a=read(d/'audit.json')
        if sha(d/'audit.json')!=s['prerequisite_audits_sha256'][v] or sha(d/'predictions.jsonl')!=a['files_sha256']['predictions.jsonl']:
            raise ValueError('Evidence changed after audit')
        variants[v]={r['image']:r for r in read_rows(d/'predictions.jsonl')}
    expected={};decisions={}
    for role,order in [('m3',cfg['m3']['candidate_order']),('mask',cfg['m3']['mask_candidate_order'])]:
        lookup={};repeats={}
        for fit,assess in [('A','B'),('B','A')]:
            choice=choose(splits['val_'+fit],variants,base,order)
            record=s['selection'][role][assess]
            if choice!=record['selected'] or record['fit_fold']!=fit:raise ValueError('Selection does not reproduce')
            for n in splits['val_'+assess]:
                lookup[n]=variants[choice][n]
                if role=='m3':repeats[n]=variants['m3-v2-repeat-g'+choice[-1]][n]
        if choose(names,variants,base,order)!=s['all_val_final_selection'][role]['selected']:
            raise ValueError('Final selection differs')
        expected[role]=[lookup[n] for n in names]
        if role=='m3':expected['repeat']=[repeats[n] for n in names]
    expected['branch-only']=[variants['m3-v2-branch-only'][n] for n in names]
    for role,rows in expected.items():
        saved=read_rows(out/(role+'.jsonl'))
        if len(saved)!=1286:raise ValueError('Incomplete crossfit')
        for actual,want in zip(saved,rows):
            if any(actual[k]!=want[k] for k in actual):raise ValueError('Selected row mismatch')
        groups={'overall':rows,'<50':[r for r in rows if r['truth']<50],
                '50-299':[r for r in rows if 50<=r['truth']<300],'>=300':[r for r in rows if r['truth']>=300]}
        for key,values in groups.items():
            for field,value in metrics(values).items():
                if not math.isclose(value,s['groups'][role]['metrics'][key][field],rel_tol=1e-12,abs_tol=1e-10):
                    raise ValueError('Crossfit metric mismatch')
        b=metrics(base_rows);m=metrics(rows);dense=metrics(groups['>=300']);bd=metrics([r for r in base_rows if r['truth']>=300])
        passed=m['mae']<=1.02*b['mae'] and m['rmse']<b['rmse'] and dense['mae']<bd['mae']
        if passed!=s['groups'][role]['candidate_gate']:raise ValueError('Admission threshold mismatch')
    write_json(out/'audit.json',{'status':'passed','n':1286,'fold_local_choices_reproduced':True,
               'selected_rows_and_metrics_reproduced':True,'summary_sha256':sha(out/'summary.json'),
               'scope':'development_only_no_test','raw_variant_audits':s['prerequisite_audits_sha256']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
