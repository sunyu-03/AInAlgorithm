import bootstrap
import argparse,json,shutil,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from assignment_env import Task
from waam_validator import run_validation,load_config
from joint_wait_search import read_baseline
from bundle_reassign36 import schedule_batch
from finite_stay import schedule_finite_stay
from run_bundle_stay import directed,export
from run_parallel import sha,write,overlap
ROOT=Path(__file__).resolve().parent
def sources():
 paths=list(ROOT.glob('*.py'))+[p for n in ('backend','algorithm') for p in (ROOT/n).rglob('*.py')]
 return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--search-input',type=Path,required=True);p.add_argument('--control-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 baseline=a.baseline.resolve();search=a.search_input.resolve();control=a.control_run.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();before=sources()
 for n in ('config.yaml','target.stl'):assert sha(baseline/n)==sha(search/n),'Search target/config mismatch'
 config=load_config(baseline/'config.yaml');robots=sorted(config.robots,key=lambda r:r.id);raw,seq,rr,trace=read_baseline(search);_,_,guard,_=read_baseline(baseline);expected=directed([guard[r] for r in range(3)]);assert directed([rr[r] for r in range(3)])==expected
 oldcontrol=json.loads((control/'run.json').read_text(encoding='utf-8'));assert oldcontrol['status']=='PASS' and not oldcontrol['source_changed_during_run'] and not oldcontrol['errors']
 assert oldcontrol['input_sha256']=={n:sha(search/n) for n in ('target.stl','config.yaml','trajectory.csv')},'Unmatched 2/4 control input'
 for n in ('bundle_reassign.py','finite_stay.py'):assert oldcontrol['source_sha256'][n]==sha(ROOT/n),'Control implementation changed'
 assert (ROOT/'bundle_reassign36.py').read_text(encoding='utf-8')==(ROOT/'bundle_reassign.py').read_text(encoding='utf-8').replace('sizes=(2,4)','sizes=(3,6)').replace('bundle_sizes=[2,4]','bundle_sizes=[3,6]')
 env=SimpleNamespace(config=config,robots=robots,homes=np.array([r.home_xyz_mm if r.home_xyz_mm is not None else r.base_xyz_mm for r in robots],float),bases=np.array([r.base_xyz_mm for r in robots],float),tasks=[Task(tuple(v['start_xyz_mm']),tuple(v['end_xyz_mm'])) for v in raw],reach_margin_mm=1.)
 report=dict(status='ERROR',input_directory=str(baseline),search_input_directory=str(search),input_sha256={n:sha(baseline/n) for n in ('target.stl','config.yaml','trajectory.csv')},search_input_sha256={n:sha(search/n) for n in ('target.stl','config.yaml','trajectory.csv')},source_sha256=before,matched_control_run=str(control),matched_control_candidates=oldcontrol['candidates'],bundle_sizes=[3,6],candidates=[],errors=[])
 try:
  guard_result=run_validation(baseline,out/'baseline_validation');report['baseline_s']=guard_result.schedule.makespan_s;choices=[]
  if guard_result.status=='PASS':choices.append((guard_result.schedule.makespan_s,baseline,'preserved_baseline',out/'baseline_validation'))
  for policy in ('bundle36','combined36'):
   try:
    if policy=='bundle36':value=schedule_batch(env,seq,rr,10.,2);batch=value
    elif 'batch' in locals():value=schedule_finite_stay(env,batch['rows'],10.)
    else:raise ValueError('Bundle construction unavailable')
    folder=out/policy;folder.mkdir();rows=value['rows'];assert directed(rows)==expected,'Directed deposition changed';write(folder/'schedule.json',value['schedule']);export(folder/'job',baseline,rows)
    result=run_validation(folder/'job',folder/'validation');report['candidates'].append(dict(policy=policy,status=result.status,makespan_s=result.schedule.makespan_s,schedule_kind=value['schedule']['kind'],**overlap(rows)))
    if result.status=='PASS' and result.schedule.makespan_s<guard_result.schedule.makespan_s-1e-6:choices.append((result.schedule.makespan_s,folder/'job',policy,folder/'validation'))
    write(out/'partial_run.json',report)
   except Exception as e:report['errors'].append(dict(policy=policy,error=f'{type(e).__name__}: {e}'));print(report['errors'][-1],flush=True)
  if not choices:raise ValueError('No full PASS solution')
  end,job,kind,validation=min(choices,key=lambda v:v[0]);dest=out/'selected';dest.mkdir()
  for n in ('target.stl','config.yaml','trajectory.csv'):shutil.copy2(job/n,dest/n)
  report.update(status='PASS',selected_s=end,selected_kind=kind,selected_validation=str(validation),reduction_percent=100*(1-end/report['baseline_s']),output_sha256={n:sha(dest/n) for n in ('target.stl','config.yaml','trajectory.csv')})
 finally:report.update(elapsed_s=time.perf_counter()-started,source_changed_during_run=before!=sources());write(out/'run.json',report)
 print(json.dumps({k:report.get(k) for k in ('status','baseline_s','candidates','selected_s','selected_kind','reduction_percent','errors')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
