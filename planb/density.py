"""M1 detection-derived density: one unit of mass for every final detection."""
import numpy as np


def boxes_to_density(boxes, height, width):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if height <= 0 or width <= 0 or not np.isfinite(boxes).all():
        raise ValueError('Invalid image size or boxes')
    density = np.zeros((height, width), dtype=np.float64)
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1+x2)/2, (y1+y2)/2
        if x2 < x1 or y2 < y1 or not (0 <= cx < width and 0 <= cy < height):
            raise ValueError('Final detection must have a valid centre inside the image')
        sx, sy = max((x2-x1)/6, .5), max((y2-y1)/6, .5)
        lo_x, hi_x = max(0, int(np.floor(cx-3*sx))), min(width, int(np.ceil(cx+3*sx))+1)
        lo_y, hi_y = max(0, int(np.floor(cy-3*sy))), min(height, int(np.ceil(cy+3*sy))+1)
        xx = (np.arange(lo_x, hi_x)-cx)/sx
        yy = (np.arange(lo_y, hi_y)-cy)/sy
        patch = np.exp(-.5*(yy[:,None]**2+xx[None,:]**2))
        density[lo_y:hi_y, lo_x:hi_x] += patch/patch.sum()
    return density


def game(density, points, level):
    """Grid Average Mean Absolute Error for one image; sum over 4**level cells."""
    h, w = density.shape
    side = 2**level
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) and not ((points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)).all():
        raise ValueError('Ground-truth points outside image')
    ys, xs = np.linspace(0,h,side+1,dtype=int), np.linspace(0,w,side+1,dtype=int)
    error = 0.
    for j in range(side):
        for i in range(side):
            truth = ((points[:,0]>=xs[i])&(points[:,0]<xs[i+1])&
                     (points[:,1]>=ys[j])&(points[:,1]<ys[j+1])).sum()
            error += abs(float(density[ys[j]:ys[j+1],xs[i]:xs[i+1]].sum())-int(truth))
    return error
