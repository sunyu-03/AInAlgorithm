"""Collinear task subdivision for reassignment experiments; geometry retained."""
import bootstrap
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from waam_validator.config.loader import load_config

def split_records(raw,maximum):
 if not np.isfinite(maximum) or maximum<=0:raise ValueError('Maximum length must be positive and finite')
 records=[];lineage=[]
 for t in raw:
  a=np.asarray(t['start_xyz_mm'],float);b=np.asarray(t['end_xyz_mm'],float)
  if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite([a,b]).all():raise ValueError('Expected finite XYZ endpoints')
  length=np.linalg.norm(b-a)
  if length<=0:raise ValueError('Zero-length deposition task')
  count=max(1,int(np.ceil(length/maximum)))
  for j in range(count):
   records.append(dict(task_id=len(records),layer=t['layer'],start_xyz_mm=(a+(b-a)*j/count).tolist(),end_xyz_mm=(a+(b-a)*(j+1)/count).tolist()))
   lineage.append(dict(task_id=len(records)-1,parent_task=t['task_id'],piece=j,pieces=count))
 return records,lineage

def main():
 p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--bead-widths',type=float,default=10.);a=p.parse_args()
 cfg=load_config(a.config);raw=json.loads(a.tasks.read_text(encoding='utf-8'));records,lineage=split_records(raw,cfg.process.bead_width_mm*a.bead_widths)
 a.out.mkdir(parents=True,exist_ok=False)
 for name,value in [('tasks.json',records),('lineage.json',lineage),('source.json',dict(source_tasks_sha256=hashlib.sha256(a.tasks.read_bytes()).hexdigest(),maximum_length_mm=cfg.process.bead_width_mm*a.bead_widths,original_task_count=len(raw),split_task_count=len(records)))]:
  (a.out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
if __name__=='__main__':main()
