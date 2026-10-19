import bootstrap
import argparse,json,time,shutil,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from assignment_env import Task
from waam_validator import run_validation,load_config
from joint_wait_search import read_baseline
from fixed_boundary_lns import directed
from template_sweep import optimize
from run_bundle_stay import export
from run_parallel import sha,write

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--budget-s',type=float,default=1200.);ap.add_argument('--seed-plan',type=Path,required=True);a=ap.parse_args();baseline=a.baseline.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 source={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')};inputs={n:sha(baseline/n) for n in ['target.stl','config.yaml','trajectory.csv']}
 report=dict(status='RUNNING',input_directory=str(baseline),input_sha256=inputs,source_sha256=source,errors=[])
 def persist():write(out/'run.json',dict(report,elapsed_s=time.perf_counter()-start))
 persist()
 try:
  cfg=load_config(baseline/'config.yaml');robots=sorted(cfg.robots,key=lambda r:r.id);raw,seq,old,_=read_baseline(baseline)
  env=SimpleNamespace(config=cfg,robots=robots,homes=np.array([r.home_xyz_mm if r.home_xyz_mm is not None else r.base_xyz_mm for r in robots]),bases=np.array([r.base_xyz_mm for r in robots]),tasks=[Task(tuple(t['start_xyz_mm']),tuple(t['end_xyz_mm'])) for t in raw],reach_margin_mm=1.)
  previous=run_validation(baseline,out/'baseline_validation');assert previous.status=='PASS';report['baseline_s']=previous.schedule.makespan_s;persist()
  rows=[old[r] for r in range(3)];plan=optimize(env,rows,a.budget_s,12,10.,seed_schedule=a.seed_plan);assert directed(plan['rows'])==directed(rows)
  write(out/'schedule.json',plan['schedule']);export(out/'candidate',baseline,plan['rows']);v=run_validation(out/'candidate',out/'candidate_validation')
  report.update(candidate_s=v.schedule.makespan_s,candidate_status=v.status,accepted_windows=plan['schedule']['accepted_count'],search_s=plan['schedule']['search_s'])
  if v.status=='PASS' and v.schedule.makespan_s<previous.schedule.makespan_s-1e-6:job=out/'candidate';end=v.schedule.makespan_s;kind='fixed_boundary_template_sweep'
  else:job=baseline;end=previous.schedule.makespan_s;kind='preserved_input'
  (out/'selected').mkdir()
  for name in inputs:shutil.copy2(job/name,out/'selected'/name)
  assert inputs=={n:sha(baseline/n) for n in inputs}
  report.update(status='PASS',selected_s=end,selected_kind=kind,saved_s=previous.schedule.makespan_s-end,trajectory_sha256=sha(out/'selected/trajectory.csv'))
 except Exception as exc:
  report.update(status='ERROR',error=str(exc),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
 finally:
  report['source_changed']=source!={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')};persist()
 print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','input_sha256']}),flush=True)
if __name__=='__main__':main()
