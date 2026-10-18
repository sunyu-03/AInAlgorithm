import bootstrap
import argparse,csv,json,shutil,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from assignment_env import Task
from waam_validator import run_validation,load_config
from joint_wait_search import read_baseline
from overlap_retimer import schedule,schedule_independent
from run_parallel import sha,write,deposition,overlap
ROOT=Path(__file__).resolve().parent
def sources():return {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if not any(v in p.parts for v in ('runs','__pycache__'))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--proof-step-mm',type=float,default=10.);p.add_argument('--independent',action='store_true');a=p.parse_args()
 if not np.isfinite(a.proof_step_mm) or a.proof_step_mm<=0:p.error('Finite positive proof step required')
 baseline=a.baseline.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);before=sources();started=time.perf_counter()
 config=load_config(baseline/'config.yaml');robots=sorted(config.robots,key=lambda r:r.id);raw,seq,oldrows,trace=read_baseline(baseline)
 env=SimpleNamespace(config=config,robots=robots,homes=np.array([r.home_xyz_mm if r.home_xyz_mm is not None else r.base_xyz_mm for r in robots],float),bases=np.array([r.base_xyz_mm for r in robots],float),tasks=[Task(tuple(v['start_xyz_mm']),tuple(v['end_xyz_mm'])) for v in raw])
 report=dict(status='ERROR',input_sha256={n:sha(baseline/n) for n in ('target.stl','config.yaml','trajectory.csv')},metadata_sha256=sha(a.metadata),source_sha256=before)
 try:
  old=run_validation(baseline,out/'baseline_validation');choices=[];report['baseline_s']=old.schedule.makespan_s
  if old.status=='PASS':choices.append((old.schedule.makespan_s,baseline,'preserved_baseline',out/'baseline_validation'))
  try:
   details=(schedule_independent if a.independent else schedule)(env,oldrows,json.loads(a.metadata.read_text(encoding='utf-8')),a.proof_step_mm)
   assert deposition(env.rows)==deposition(list(oldrows.values())),'Deposition changed'
   job=out/'job';job.mkdir()
   for n in ('target.stl','config.yaml'):shutil.copy2(baseline/n,job/n)
   with (job/'trajectory.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['robot_id','time_s','x_mm','y_mm','z_mm','mode'])
    for r,rr in enumerate(env.rows):writer.writerows([[r+1,*v] for v in rr])
   write(out/'schedule.json',details);new=run_validation(job,out/'validation');report.update(candidate_status=new.status,candidate_s=new.schedule.makespan_s,**overlap(env.rows))
   if new.status=='PASS' and new.schedule.makespan_s<old.schedule.makespan_s-1e-6:choices.append((new.schedule.makespan_s,job,details['kind'],out/'validation'))
  except Exception as e:report.update(candidate_status='ERROR',candidate_error=f'{type(e).__name__}: {e}')
  if not choices:raise ValueError('No full PASS solution')
  end,job,kind,validation=min(choices,key=lambda v:v[0]);dest=out/'selected';dest.mkdir()
  for n in ('target.stl','config.yaml','trajectory.csv'):shutil.copy2(job/n,dest/n)
  report.update(status='PASS',selected_s=end,selected_kind=kind,selected_validation=str(validation),reduction_percent=100*(1-end/old.schedule.makespan_s),output_sha256={n:sha(dest/n) for n in ('target.stl','config.yaml','trajectory.csv')})
 finally:report.update(elapsed_s=time.perf_counter()-started,source_changed_during_run=before!=sources());write(out/'run.json',report)
 print(json.dumps({k:report.get(k) for k in ('status','candidate_status','candidate_s','baseline_s','selected_s','reduction_percent','candidate_error')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
