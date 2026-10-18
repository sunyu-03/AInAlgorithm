import bootstrap
from pathlib import Path
import argparse,json,hashlib,csv,math,time,shutil
import numpy as np
from algorithm.environment.waam_env import WAAMBaselineEnv
from parallel_scheduler import resident,pipeline
from waam_validator import run_validation

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def deposition(rows):
 result=[]
 for own in rows:
  for a,b in zip(own,own[1:]):
   if a[4]=='D':
    ends=sorted((tuple(np.asarray(a[1:4],np.float32)),tuple(np.asarray(b[1:4],np.float32))))
    result.append(tuple(float(v) for p in ends for v in p))
 return sorted(result)
def overlap(rows):
 events=[]
 for r,own in enumerate(rows):
  for a,b in zip(own,own[1:]):
   if a[4]!='W':events.extend([(a[0],r,a[4],1),(b[0],r,a[4],-1)])
 active={};last=0.;both=dd=0.
 for t,r,mode,sign in sorted(events,key=lambda e:(e[0],e[3],e[1],e[2])):
  dt=t-last
  if len(active)>=2:both+=dt
  if sum(v=='D' for v in active.values())>=2:dd+=dt
  if sign>0:active[r]=mode
  else:active.pop(r,None)
  last=t
 return dict(simultaneous_motion_s=both,simultaneous_deposition_s=dd)
def main():
 p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--proof-step-mm',type=float,default=25.);p.add_argument('--cycle-test',action='store_true');p.add_argument('--grouped-only',action='store_true')
 a=p.parse_args()
 if not math.isfinite(a.proof_step_mm) or a.proof_step_mm<=0:p.error('proof step must be positive and finite')
 job=a.job.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);baseline=a.baseline.resolve()
 for n in ('config.yaml','target.stl'):
  if sha(job/n)!=sha(baseline/n):raise ValueError('Baseline must use identical original inputs')
 raw=json.loads(a.tasks.read_text(encoding='utf-8'));expected=sorted(tuple(float(v) for q in sorted((tuple(np.asarray(t['start_xyz_mm'],np.float32)),tuple(np.asarray(t['end_xyz_mm'],np.float32)))) for v in q) for t in raw)
 sources={str(x.relative_to(ROOT)):sha(x) for x in ROOT.rglob('*.py') if not any(v in x.parts for v in ('runs','__pycache__'))};started=time.perf_counter();cases=[]
 methods=[('resident',resident),('pipeline',pipeline)]
 if a.cycle_test:methods.append(('pipeline_cycle',lambda env,step:pipeline(env,step,True)))
 if a.grouped_only:methods=[('pipeline_grouped',lambda env,step:pipeline(env,step,grouped=True))]
 for label,fn in methods:
  folder=out/label;folder.mkdir();env=None
  try:
   env=WAAMBaselineEnv(dict(job_dir=str(job),stl_path=str(job/'target.stl'),config_path=str(job/'config.yaml'),tasks=raw,reach_margin_mm=1.))
   details=fn(env,a.proof_step_mm)
   if deposition(env.rows)!=expected:raise ValueError('Deposition segment multiset changed')
   if sorted(t['task'] for t in details['trace'])!=list(range(len(raw))):raise ValueError('Tasks duplicated or omitted')
   details.update(overlap(env.rows));write(folder/'schedule.json',details);env.export(folder/'job')
   result=run_validation(folder/'job',folder/'validation')
   cases.append(dict(kind=label,status=result.status,makespan_s=result.schedule.makespan_s,job=str(folder/'job'),shape=result.summary_dict()['shape'],geometry_unchanged=True,**overlap(env.rows)))
  except Exception as e:cases.append(dict(kind=label,status='ERROR',error=f'{type(e).__name__}: {e}'))
  finally:
   if env is not None:env.close()
  write(out/'candidates.json',cases)
 # Freshly validate the preserved baseline as well.
 old=run_validation(baseline,out/'baseline_validation');cases.append(dict(kind='baseline',status=old.status,makespan_s=old.schedule.makespan_s,job=str(baseline),shape=old.summary_dict()['shape']))
 passes=[c for c in cases if c['status']=='PASS']
 if not passes:raise RuntimeError('No complete PASS trajectory')
 best=min(passes,key=lambda c:c['makespan_s']);selected=out/'selected';selected.mkdir()
 for n in ('target.stl','config.yaml','trajectory.csv'):shutil.copy2(Path(best['job'])/n,selected/n)
 if any(sha(selected/n)!=sha(Path(best['job'])/n) for n in ('target.stl','config.yaml','trajectory.csv')):raise ValueError('Selected copy hash mismatch')
 after={str(x.relative_to(ROOT)):sha(x) for x in ROOT.rglob('*.py') if not any(v in x.parts for v in ('runs','__pycache__'))}
 report=dict(status='PASS',selected=best,candidates=cases,baseline_s=old.schedule.makespan_s,reduction_percent=100*(1-best['makespan_s']/old.schedule.makespan_s),input_sha256={n:sha(job/n) for n in ('target.stl','config.yaml')},tasks_sha256=sha(a.tasks),output_sha256={n:sha(selected/n) for n in ('target.stl','config.yaml','trajectory.csv')},source_sha256=sources,source_changed_during_run=sources!=after,elapsed_s=time.perf_counter()-started)
 write(out/'run.json',report);print(json.dumps(report['selected'],ensure_ascii=False),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
