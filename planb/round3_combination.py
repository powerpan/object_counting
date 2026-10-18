"""Assemble the registered no-missing-input, no-calibration combination."""
import numpy as np
from planb.density import boxes_to_density,points_for_game,game
from planb.round2.spatial import cell_mass,point_cells,localization


def ready(plan):
    if plan['missing_inference_variants'] or plan['requires_calibration_refit']:
        raise RuntimeError('Combination requires an additional preregistered execution entry')
    if any(d['calibration']['admitted'] for d in plan['decisions'].values()):
        raise ValueError('Calibration flag inconsistent')


def density_from_source(row,component):
    h,w=row['height'],row['width']
    if component in ('T2','T4'):
        views=row['detail']['views'];expected=2 if component=='T2' else 4
        if len(views)!=expected:raise ValueError('Wrong TTA view count')
        maps=[];counts=[]
        for v in views:
            if len(v['boxes_original_xyxy'])!=v['count']:raise ValueError('Wrong view count')
            maps.append(boxes_to_density(v['boxes_original_xyxy'],h,w));counts.append(v['count'])
        if float(np.mean(counts))!=row['prediction']:raise ValueError('TTA average count differs')
        density=np.stack(maps).mean(axis=0,dtype=np.float64)
    elif component in ('B0','m3-v2-g2','m3-v2-g3','b0-refine-on','b0-refine-off'):
        if len(row['boxes_xyxy'])!=row['prediction']:raise ValueError('Detector count differs')
        density=boxes_to_density(row['boxes_xyxy'],h,w)
    else:raise ValueError('Missing or unregistered component')
    if not np.isfinite(density).all() or abs(float(density.sum())-row['prediction'])>1e-6:
        raise ValueError('Combination density mass differs')
    return density


def assemble(row,component,annotation,source_sha):
    h,w=row['height'],row['width'];density=density_from_source(row,component)
    points=points_for_game(annotation['points'],h,w)
    if len(points)!=row['truth']:raise ValueError('Source truth differs')
    exemplars=np.asarray(annotation['box_examples_coordinates'],dtype=np.float32)[:3,[0,2],:].reshape(-1,4)
    mass=cell_mass(density);_,truth8=point_cells(points,h,w)
    games={f'game{level}':game(density,points,level) for level in (1,2)}
    for key,value in games.items():
        if not np.isclose(value,row[key],rtol=1e-12,atol=1e-8):raise ValueError('Source GAME differs')
    if 'mass8' in row and not np.allclose(mass,row['mass8'],rtol=1e-12,atol=1e-8):raise ValueError('Source cell mass differs')
    location=localization(row['boxes_xyxy'],points,exemplars)
    if 'localization' in row and location!=row['localization']:raise ValueError('Source localization differs')
    seconds=row['inference_seconds_per_image']
    if not np.isfinite(seconds) or seconds<=0:raise ValueError('Invalid component timing')
    return {'image':row['image'],'truth':row['truth'],'prediction':row['prediction'],'width':w,'height':h,
            'density_mass':float(density.sum()),'mass8':mass.tolist(),'truth8':truth8.tolist(),**games,
            'localization':location,'inference_seconds_per_image':seconds,'source_variant':component,
            'source_predictions_sha256':source_sha,
            'source_density_file_sha256':row.get('density_file_sha256'),
            'timing_scope':'selected saved component, not measured end-to-end combination'}
