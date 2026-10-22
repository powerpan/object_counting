"""Image rendering and six-click exemplar selection for the demonstration."""
import numpy as np
from PIL import Image,ImageDraw


def selection(points,width,height):
    if len(points)>6:raise ValueError('Only three boxes are allowed')
    boxes=[]
    for i in range(0,len(points)-1,2):
        a,b=points[i:i+2];x0,x1=sorted([float(a[0]),float(b[0])]);y0,y1=sorted([float(a[1]),float(b[1])])
        if x1-x0<1 or y1-y0<1 or x0<0 or y0<0 or x1>width or y1>height:raise ValueError('Invalid exemplar rectangle')
        boxes.append([x0,y0,x1,y1])
    return boxes


def boxes_image(image,boxes,color=(255,75,75),prompts=()):
    im=Image.fromarray(np.asarray(image,dtype=np.uint8)).convert('RGB');draw=ImageDraw.Draw(im)
    for b in boxes:draw.rectangle(tuple(map(float,b)),outline=color,width=2)
    for i,b in enumerate(prompts):
        draw.rectangle(tuple(map(float,b)),outline=(0,220,255),width=3);draw.text((b[0]+3,b[1]+2),str(i+1),fill=(0,220,255))
    return np.asarray(im)


def density_image(image,density):
    d=np.asarray(density,dtype=float);scale=float(d.max())
    strength=np.sqrt(d/max(scale,1e-15));alpha=.65*strength[...,None]
    rgb=np.zeros((*d.shape,3));rgb[...,0]=255;rgb[...,1]=210*(1-strength)
    return np.clip(np.asarray(image)*(1-alpha)+rgb*alpha,0,255).astype(np.uint8)
