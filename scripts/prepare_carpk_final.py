"""Freeze author archive bytes and prompt handling before CARPK inference."""
import argparse
from pathlib import Path
from PIL import Image
from final_common import sha,write


def prompts(rows):
    if len(rows)<2:raise ValueError('Unexpected CARPK image with fewer than two cars')
    boxes=[r[:4] for r in rows[:3]]
    return boxes+[boxes[-1]]*(3-len(boxes))


def main(data,output):
    if output.exists():raise FileExistsError('Frozen CARPK manifest exists')
    ids=(data/'ImageSets/test.txt').read_text().splitlines()
    if len(ids)!=459 or len(set(ids))!=459:raise ValueError('Wrong official test split')
    files=[]
    for key in ids:
        image=data/'Images'/(key+'.png');ann=data/'Annotations'/(key+'.txt')
        rows=[list(map(int,line.split())) for line in ann.read_text().splitlines() if line.strip()]
        with Image.open(image) as im:w,h=im.size
        if any(len(r)!=5 or r[4]!=1 or not (0<=r[0]<r[2]<=w and 0<=r[1]<r[3]<=h) for r in rows):
            raise ValueError('Invalid official annotation')
        files.append({'image':image.name,'class':'car','width':w,'height':h,'truth':len(rows),
            'image_path':'data/carpk/Images/'+image.name,'image_sha256':sha(image),
            'annotation_path':'data/carpk/Annotations/'+ann.name,'annotation_sha256':sha(ann),
            'exemplars':prompts(rows),'distinct_exemplars':min(3,len(rows)),
            'points':[[(r[0]+r[2])/2,(r[1]+r[3])/2] for r in rows]})
    write(output,{'version':1,'dataset':'CARPK','split':'test','license_confirmed':True,
        'author_page':'https://lafi.github.io/LPN/',
        'official_archive_sha256':'d533004d89a5b7fe32c39686dd5237c85693484686a6d886a03d8d86d2057266',
        'official_test_split_sha256':sha(data/'ImageSets/test.txt'),
        'prompt_rule':'First three official boxes in file order. For the five two-car images, repeat the second box as slot three.',
        'ground_truth_rule':'All official vehicle boxes; centers for spatial metrics; repeated prompt does not duplicate truth.',
        'ids':[r['image'] for r in files],'files':files})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.data,a.output)
