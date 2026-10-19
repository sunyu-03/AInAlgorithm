import bootstrap
from pathlib import Path
import argparse,json,time,shutil
from algorithm.environment.waam_env import WAAMBaselineEnv
from joint_wait_search import read_baseline,schedule
from run_parallel import sha,write,deposition,overlap
from waam_validator import run_validation
ROOT=Path(__file__).resolve().parent
def sources():return {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if not any(x in p.parts for x in ('runs','__pycache__'))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--metadata',type=Path);p.add_argument('--proof-step-mm',type=float,default=10.);p.add_argument('--candidates',type=int,default=3);a=p.parse_args()
 if a.proof_step_mm<=0 or a.candidates<1:p.error('Positive step and candidate count required')
 baseline=a.baseline.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);before=sources();started=time.perf_counter()
 report=dict(status='ERROR',input_sha256={n:sha(baseline/n) for n in ('target.stl','config.yaml','trajectory.csv')},source_sha256=before)
 raw,seq,oldrows,trace=read_baseline(baseline);write(out/'tasks.json',raw);env=WAAMBaselineEnv(dict(job_dir=str(baseline),stl_path=str(baseline/'target.stl'),config_path=str(baseline/'config.yaml'),tasks=raw,reach_margin_mm=1.))
 try:
  old=run_validation(baseline,out/'baseline_validation');report['baseline_s']=old.schedule.makespan_s;choices=[]
  if old.status=='PASS':choices.append((old.schedule.makespan_s,baseline,'preserved_baseline',out/'baseline_validation'))
  try:
   metadata=json.loads(a.metadata.read_text(encoding='utf-8')) if a.metadata else None
   details=schedule(env,seq,oldrows,metadata,a.proof_step_mm,a.candidates)
   assert deposition(env.rows)==deposition([oldrows[r] for r in range(3)]),'Deposition changed'
   assert sorted((v['task'],v['robot_id']) for v in details['trace'])==sorted((v['task'],v['robot_id']) for v in trace),'Assignment changed'
   write(out/'schedule.json',details);env.export(out/'job');new=run_validation(out/'job',out/'validation')
   report.update(candidate_status=new.status,candidate_s=new.schedule.makespan_s,geometry_unchanged=True,**overlap(env.rows))
   if new.status=='PASS':choices.append((new.schedule.makespan_s,out/'job','joint_wait_search',out/'validation'))
  except Exception as e:report.update(candidate_status='ERROR',candidate_error=f'{type(e).__name__}: {e}')
  if not choices:raise ValueError('No full PASS path')
  end,job,kind,validation=min(choices,key=lambda v:v[0]);dest=out/'selected';dest.mkdir()
  for n in ('target.stl','config.yaml','trajectory.csv'):shutil.copy2(job/n,dest/n)
  report.update(status='PASS',selected_s=end,selected_kind=kind,selected_validation=str(validation),reduction_percent=100*(1-end/old.schedule.makespan_s),output_sha256={n:sha(dest/n) for n in ('target.stl','config.yaml','trajectory.csv')})
 finally:
  env.close();report.update(elapsed_s=time.perf_counter()-started,source_changed_during_run=before!=sources());write(out/'run.json',report)
 print(json.dumps({k:report.get(k) for k in ('status','candidate_status','candidate_s','baseline_s','selected_s','reduction_percent','candidate_error')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
