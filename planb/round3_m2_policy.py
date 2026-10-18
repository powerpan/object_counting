"""Three registered M2-v2 changes; pure count/geometry policy without labels."""
import math
import numpy as np


def select_candidates(boxes,exemplars,relative,cosine,config):
    b=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    e=np.asarray(exemplars,dtype=np.float64).reshape(-1,4)
    r=np.asarray(relative,dtype=np.float64);c=np.asarray(cosine,dtype=np.float64)
    if len(e)!=3 or r.shape!=(len(b),) or c.shape!=(len(b),):raise ValueError('Invalid candidate shapes')
    if not all(np.isfinite(x).all() for x in (b,e,r,c)):raise ValueError('Nonfinite candidate inputs')
    sides=b[:,2:]-b[:,:2];es=e[:,2:]-e[:,:2]
    if (es<=0).any() or (sides<0).any():raise ValueError('Invalid candidate boxes')
    centers=(b[:,:2]+b[:,2:])/2
    inside=((centers[None,:,0]>=b[:,None,0])&(centers[None,:,0]<=b[:,None,2])&
            (centers[None,:,1]>=b[:,None,1])&(centers[None,:,1]<=b[:,None,3]))
    # Every valid rectangle contains its own center; exclude it exactly once.
    others=inside.sum(axis=1)-1
    scales=np.sqrt(np.prod(sides,axis=1));orig=np.sqrt(np.prod(es,axis=1))
    ratios=np.maximum(scales[:,None]/orig,orig/np.maximum(scales[:,None],1e-12))
    eligible=((sides>=config['min_side_in_model_pixels']).all(axis=1)&
              (ratios.min(axis=1)>=config['min_scale_ratio'])&
              (r>=config['min_relative_objectness'])&(c>=config['min_mean_cosine']))
    rejected=eligible&(others>=config['other_centers_rejection_minimum'])
    valid=eligible&~rejected
    chosen=[];bins=set()
    for i in np.argsort(-(r*c),kind='stable'):
        if not valid[i]:continue
        bucket=math.floor(math.log(scales[i]/np.median(orig),config['min_scale_ratio']))
        if bucket in bins:continue
        bins.add(bucket);chosen.append(int(i))
        if len(chosen)==config['max_extra_exemplars']:break
    return {'selected_indices':chosen,'other_centers':others.astype(int).tolist(),
            'eligible_before_single_object_filter':np.flatnonzero(eligible).tolist(),
            'rejected_by_single_object_filter':np.flatnonzero(rejected).tolist()}


def should_fallback(first_count,second_count,threshold=.5):
    if not isinstance(first_count,(int,np.integer)) or not isinstance(second_count,(int,np.integer)) or min(first_count,second_count)<0:
        raise ValueError('Counts must be nonnegative integers')
    return bool(abs(second_count-first_count)/max(first_count,1)>threshold)
