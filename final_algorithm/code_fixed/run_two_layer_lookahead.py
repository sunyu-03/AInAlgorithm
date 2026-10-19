import bootstrap
from pathlib import Path
import argparse,csv,json,shutil,time
from types import SimpleNamespace
import numpy as np
from assignment_env import Task
from waam_validator import run_validation,load_config
from joint_wait_search import read_baseline
from two_layer_lookahead import schedule_both
from run_parallel import sha,write,deposition,overlap
ROOT=Path(__file__).resolve().parent
def sources():return {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if not any(v in p.parts for v in ('runs','__pycache__'))}
def export(job,baseline,rows):
 job.mkdir()
 for n in ('target.stl','config.yaml'):shutil.copy2(baseline/n,job/n)
 with (job/'trajectory.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['robot_id','time_s','x_mm','y_mm','z_mm','mode'])
  for r,rr in enumerate(rows):w.writerows([[r+1,*v] for v in rr])
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--proof-step-mm',type=float,default=10.);p.add_argument('--beam',type=int,default=2);a=p.parse_args()
 if not np.isfinite(a.proof_step_mm) or a.proof_step_mm<=0 or a.beam<1 or a.beam>6:p.error('Positive finite step and beam 1..6 required')
 baseline=a.baseline.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();before=sources()
 config=load_config(baseline/'config.yaml');robots=sorted(config.robots,key=lambda r:r.id);raw,seq,oldrows,trace=read_baseline(baseline)
 env=SimpleNamespace(config=config,robots=robots,homes=np.array([r.home_xyz_mm if r.home_xyz_mm is not None else r.base_xyz_mm for r in robots],float),bases=np.array([r.base_xyz_mm for r in robots],float),tasks=[Task(tuple(v['start_xyz_mm']),tuple(v['end_xyz_mm'])) for v in raw],reach_margin_mm=1.)
 report=dict(status='ERROR',input_directory=str(baseline),input_sha256={n:sha(baseline/n) for n in ('target.stl','config.yaml','trajectory.csv')},source_sha256=before,candidates=[])
 try:
  old=run_validation(baseline,out/'baseline_validation');report['baseline_s']=old.schedule.makespan_s;choices=[]
  if old.status=='PASS':choices.append((old.schedule.makespan_s,baseline,'preserved_baseline',out/'baseline_validation'))
  try:
   outputs=schedule_both(env,seq,oldrows,a.proof_step_mm,a.beam)
   for policy,value in outputs.items():
    folder=out/policy;folder.mkdir();rows=value['rows'];assert deposition(rows)==deposition(list(oldrows.values())),'Deposition changed'
    write(folder/'schedule.json',value['schedule']);export(folder/'job',baseline,rows);result=run_validation(folder/'job',folder/'validation')
    record=dict(policy=policy,status=result.status,makespan_s=result.schedule.makespan_s,**overlap(rows));report['candidates'].append(record)
    if result.status=='PASS' and result.schedule.makespan_s<old.schedule.makespan_s-1e-6:choices.append((result.schedule.makespan_s,folder/'job',policy,folder/'validation'))
  except Exception as e:report['candidate_error']=f'{type(e).__name__}: {e}'
  if not choices:raise ValueError('No full PASS path')
  end,job,kind,validation=min(choices,key=lambda v:v[0]);dest=out/'selected';dest.mkdir()
  for n in ('target.stl','config.yaml','trajectory.csv'):shutil.copy2(job/n,dest/n)
  report.update(status='PASS',selected_s=end,selected_kind=kind,selected_validation=str(validation),reduction_percent=100*(1-end/old.schedule.makespan_s),output_sha256={n:sha(dest/n) for n in ('target.stl','config.yaml','trajectory.csv')})
 finally:report.update(elapsed_s=time.perf_counter()-started,source_changed_during_run=before!=sources());write(out/'run.json',report)
 print(json.dumps({k:report.get(k) for k in ('status','baseline_s','candidates','selected_s','selected_kind','reduction_percent','candidate_error')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
