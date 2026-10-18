"""Single-owner stage budgets. No carry-over or implicit retry."""
import json
import math
from pathlib import Path


class StageLedger:
    def __init__(self,path):
        self.path=Path(path);self.data=json.loads(self.path.read_text())
        self.validate()

    def validate(self):
        d=self.data
        if d['version']!='round3-ledger-v1' or sum(d['caps'].values())!=d['total_cap']:
            raise ValueError('Invalid stage budgets')
        ids=[e['id'] for e in d['entries']]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate budget entry')
        for e in d['entries']:
            if e['stage'] not in d['caps'] or not math.isfinite(e['seconds']) or e['seconds']<0:
                raise ValueError('Invalid settled time')

    @classmethod
    def create(cls,path,caps,total):
        with Path(path).open('x') as s:
            json.dump({'version':'round3-ledger-v1','caps':caps,'total_cap':total,
                       'entries':[],'active':None,'state':'open'},s,indent=2)
        return cls(path)

    def save(self):
        temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(self.data,indent=2)+'\n');temp.replace(self.path)

    def remaining(self,stage):
        return min(self.data['caps'][stage]-sum(e['seconds'] for e in self.data['entries'] if e['stage']==stage),
                   self.data['total_cap']-sum(e['seconds'] for e in self.data['entries']))

    def begin(self,identifier,stage,cap,utc):
        if self.data['state']!='open' or self.data['active'] is not None:
            raise RuntimeError('Unsettled or stopped work')
        if identifier in {e['id'] for e in self.data['entries']}:
            raise FileExistsError('Repeated stage execution')
        limit=min(float(cap),self.remaining(stage))
        if not math.isfinite(limit) or limit<=10:raise TimeoutError('Stage budget exhausted')
        self.data['active']={'id':identifier,'stage':stage,'limit':limit,'started_utc':utc};self.save()
        return limit

    def finish(self,identifier,elapsed,passed,utc):
        a=self.data['active']
        if a is None or a['id']!=identifier:raise ValueError('Settlement mismatch')
        if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid elapsed time')
        accepted=bool(passed) and elapsed<=a['limit'] and elapsed<=self.remaining(a['stage'])
        self.data['entries'].append({**a,'seconds':elapsed,'passed':accepted,'finished_utc':utc})
        self.data['active']=None
        if not accepted:self.data['state']='stopped'
        self.save();return accepted
