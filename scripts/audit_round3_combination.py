"""Reconstruct combination lineage, density mass and reported statistics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_round3_combination import load_inputs
from round3_calibration_io import read,sha,write,read_rows,utc
from audit_round3_calibration import check_tree,metric,passes
from audit_round3_m3 import report
from audit_round3_tta import rebuild_view
from planb.density import boxes_to_density,points_for_game,game
from planb.round2.spatial import cell_mass,point_cells,localization


def close(a,b):
    if not np.allclose(a,b,rtol=1e-12,atol=1e-8):raise ValueError('Combination numerical reconstruction differs')


def check_lineage(row,source,variant,digest):
    if row['source_variant']!=variant or row['source_predictions_sha256']!=digest:
        raise ValueError('Combination selected an unplanned source')
    for key in ('image','truth','prediction','width','height','inference_seconds_per_image'):
        if row[key]!=source[key]:raise ValueError('Selected row differs from immutable source')
    if row['source_density_file_sha256']!=source.get('density_file_sha256'):
        raise ValueError('Density provenance differs')


def main(root,asset_root,receipt_path):
    cfg,receipt,names,labels,base,source,annotations,plan,routes,proof=load_inputs(root,asset_root,receipt_path)
    out=root/'results/round3-combination-v1/assembly-v1'
    if (out/'audit.json').exists():raise FileExistsError('Combination audit already exists')
    protocol=read(out/'protocol.json');summary=read(out/'summary.json');rows=read_rows(out/'predictions.jsonl')
    if protocol['files_sha256']!=receipt['files_sha256'] or any(protocol[k]!=v for k,v in proof.items()):
        raise ValueError('Combination source or input changed')
    if summary['status']!='complete' or summary['n']!=1286 or [r['image'] for r in rows]!=names:
        raise ValueError('Incomplete combination')
    max_error=0.
    for row in rows:
        n=row['image'];variant=routes[n];original=source[variant][n]
        check_lineage(row,original,variant,proof['selected_source_predictions_sha256'][variant])
        h,w=row['height'],row['width'];points=points_for_game(annotations[n]['points'],h,w)
        if len(points)!=row['truth']:raise ValueError('Official truth differs')
        if variant in ('T2','T4'):
            maps=[rebuild_view(v,h,w)[0] for v in original['detail']['views']]
            density=np.zeros((h,w),dtype=np.float64)
            for d in maps:density+=d
            density/=len(maps)
        else:density=boxes_to_density(original['boxes_xyxy'],h,w)
        close(density.sum(),row['density_mass']);close(cell_mass(density),row['mass8'])
        _,truth=point_cells(points,h,w)
        if not np.array_equal(truth,row['truth8']):raise ValueError('Cell point count differs')
        max_error=max(max_error,abs(float(density.sum())-row['prediction']))
        if max_error>1e-6:raise ValueError('Density mass differs from reported count')
        for level in (1,2):close(game(density,points,level),row['game'+str(level)])
        exemplars=np.asarray(annotations[n]['box_examples_coordinates'],dtype=np.float32)[:3,[0,2],:].reshape(-1,4)
        if localization(original['boxes_xyxy'],points,exemplars)!=row['localization']:
            raise ValueError('Localization differs')
    groups={'overall':rows,'<50':[r for r in rows if r['truth']<50],
            '50-299':[r for r in rows if 50<=r['truth']<300],'>=300':[r for r in rows if r['truth']>=300]}
    for key,values in groups.items():
        for field,value in metric(values).items():close(value,summary['result']['metrics'][key][field])
    if passes(rows,base)!=summary['result']['candidate_gate']:raise ValueError('Combined gate differs')
    check_tree(summary['result'],report(rows,base,labels,cfg['bootstrap_replicates']))
    expected_counts={v:sum(x==v for x in routes.values()) for v in sorted(set(routes.values()))}
    if summary['routing_counts']!=expected_counts or summary['all_val_future_decision']!=plan['decisions']['future-only']:
        raise ValueError('Decision report differs')
    write(out/'audit.json',{'status':'passed','n':len(rows),'verified_utc':utc(),
          'selected_sources_geometry_mass_game_localization_reconstructed':True,
          'count_metrics_and_gate_independently_recomputed':True,
          'paired_intervals_recomputed_with_registered_statistics':True,
          'maximum_map_mass_error':max_error,'scope':'development_only_no_test','gpu_seconds':0,
          'files_sha256':{n:sha(out/n) for n in ('protocol.json','summary.json','predictions.jsonl')}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.asset_root.resolve(),a.receipt.resolve())
