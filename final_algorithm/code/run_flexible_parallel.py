import bootstrap
from pathlib import Path
import argparse,json,hashlib,shutil,time
from algorithm.environment.waam_env import WAAMBaselineEnv
from run_parallel import deposition,overlap,write,sha
from flexible_parallel import schedule
from waam_validator import run_validation
import numpy as np

ROOT=Path(__file__).resolve().parent
def sources():return {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if not any(x in p.parts for x in ('runs','__pycache__'))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--tasks',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--proof-step-mm',type=float,default=25.);p.add_argument('--no-reversal',action='store_true');a=p.parse_args()
 if not np.isfinite(a.proof_step_mm) or a.proof_step_mm<=0:p.error('proof step must be positive and finite')
 job=a.job.resolve();baseline=a.baseline.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 if any(sha(job/n)!=sha(baseline/n) for n in ('target.stl','config.yaml')):raise ValueError('Baseline inputs differ')
 raw=json.loads(a.tasks.read_text(encoding='utf-8'));expected=sorted(tuple(float(v) for q in sorted((tuple(np.asarray(t['start_xyz_mm'],np.float32)),tuple(np.asarray(t['end_xyz_mm'],np.float32)))) for v in q) for t in raw)
 before=sources();started=time.perf_counter();env=None;report=dict(status='ERROR',input_sha256={n:sha(job/n) for n in ('target.stl','config.yaml')},source_sha256=before,tasks_sha256=sha(a.tasks))
 try:
  env=WAAMBaselineEnv(dict(job_dir=str(job),stl_path=str(job/'target.stl'),config_path=str(job/'config.yaml'),tasks=raw,reach_margin_mm=1.))
  details=schedule(env,a.proof_step_mm,not a.no_reversal)
  assert deposition(env.rows)==expected,'Deposition changed'
  assert sorted(t['task'] for t in details['trace'])==list(range(len(raw))),'Task omission or duplication'
  details.update(overlap(env.rows));write(out/'schedule.json',details);env.export(out/'job')
  result=run_validation(out/'job',out/'validation');report.update(candidate_status=result.status,candidate_s=result.schedule.makespan_s,geometry_unchanged=True,**overlap(env.rows))
  old=run_validation(baseline,out/'baseline_validation');report['baseline_s']=old.schedule.makespan_s
  choices=[]
  if result.status=='PASS':choices.append((result.schedule.makespan_s,out/'job','flexible_parallel',out/'validation'))
  if old.status=='PASS':choices.append((old.schedule.makespan_s,baseline,'preserved_baseline',out/'baseline_validation'))
  if not choices:raise ValueError('No complete PASS candidate')
  end,selected,kind,validation=min(choices,key=lambda x:x[0]);dest=out/'selected';dest.mkdir()
  for n in ('target.stl','config.yaml','trajectory.csv'):shutil.copy2(selected/n,dest/n)
  if any(sha(dest/n)!=sha(selected/n) for n in ('target.stl','config.yaml','trajectory.csv')):raise ValueError('Selected copy mismatch')
  report.update(status='PASS',selected_kind=kind,selected_s=end,selected_validation=str(validation),reduction_percent=100*(1-end/old.schedule.makespan_s),output_sha256={n:sha(dest/n) for n in ('target.stl','config.yaml','trajectory.csv')})
 except Exception as e:report['error']=f'{type(e).__name__}: {e}';raise
 finally:
  if env is not None:env.close()
  report.update(elapsed_s=time.perf_counter()-started,source_changed_during_run=before!=sources());write(out/'run.json',report)
 print(json.dumps({k:report.get(k) for k in ('status','candidate_s','selected_s','baseline_s','reduction_percent','simultaneous_deposition_s')},ensure_ascii=False),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
