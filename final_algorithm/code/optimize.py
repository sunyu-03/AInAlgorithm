"""STL/config -> speed candidates -> unchanged official validator -> fastest PASS."""
import bootstrap
from pathlib import Path
import argparse,json,hashlib,shutil,sys,time,importlib.metadata
from algorithm.environment.waam_env import WAAMBaselineEnv
from algorithm.proven_segments import schedule_proven_segments
from waam_validator import run_validation
from speed_tasks import generate

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def sources():return {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if not any(x in p.parts for x in ('runs','__pycache__'))}

def incumbents(job,archive):
 matches=[];wanted={n:sha(job/n) for n in ('target.stl','config.yaml')}
 for p in sorted(archive.rglob('summary.json')):
  try:
   s=read(p);v=read(p.parent/'validation_inputs.json')['inputs']
   old=Path(s['input']['directory'])
   if s['status']!='PASS' or not all(v[n]['sha256']==wanted[n] and sha(old/n)==wanted[n] for n in wanted):continue
   if sha(old/'trajectory.csv')!=v['trajectory.csv']['sha256']:continue
   matches.append(dict(kind='preserved_PASS',makespan_s=s['schedule']['makespan_s'],job=str(old),summary=str(p),csv_sha256=v['trajectory.csv']['sha256']))
  except (OSError,KeyError,ValueError):continue
 return sorted(matches,key=lambda r:r['makespan_s'])

def candidate(job,out,records,label):
 folder=out/label;folder.mkdir();env=None
 try:
  write(folder/'tasks.json',[r.to_dict() for r in records])
  env=WAAMBaselineEnv(dict(job_dir=str(job),stl_path=str(job/'target.stl'),config_path=str(job/'config.yaml'),tasks=[r.to_dict() for r in records],reach_margin_mm=1.))
  schedule_proven_segments(env);env.export(folder/'job')
  write(folder/'schedule.json',env.segment_diagnostics)
  result=run_validation(folder/'job',folder/'validation')
  return dict(kind=label,status=result.status,makespan_s=result.schedule.makespan_s,job=str(folder/'job'),shape=result.summary_dict()['shape'])
 except Exception as e:
  d=dict(kind=label,status='ERROR',error=f'{type(e).__name__}: {e}');write(folder/'error.json',d);return d
 finally:
  if env:env.close()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--archive',type=Path,default=ROOT.parent/'waam_generalization/integration/evidence');p.add_argument('--reuse-tasks',type=Path)
 p.add_argument('--skip-geometry',action='store_true');a=p.parse_args();job=a.job.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 initial=sources();started=time.perf_counter();old=incumbents(job,a.archive);write(out/'incumbents.json',old)
 report=dict(objective='minimize makespan subject to full official PASS',input_sha256={n:sha(job/n) for n in ('target.stl','config.yaml')},source_sha256=initial,candidates=[],incumbent_s=old[0]['makespan_s'] if old else None)
 try:
  if not a.skip_geometry:
   try:
    records,pre=generate(job);write(out/'shape_budget.json',pre);report['candidates'].append(candidate(job,out,records,'shape_budget'))
   except Exception as e:report['candidates'].append(dict(kind='shape_budget',status='ERROR',error=str(e)))
  if a.reuse_tasks:
   from algorithm.preprocessing.task_generator import DepositionTask
   import numpy as np
   data=read(a.reuse_tasks);records=[DepositionTask(t['task_id'],t['layer'],np.array(t['start_xyz_mm']),np.array(t['end_xyz_mm'])) for t in data]
   report['candidates'].append(candidate(job,out,records,'legacy_parallel'))
  pool=[r for r in report['candidates'] if r['status']=='PASS']+old
  if not pool:raise RuntimeError('No full PASS candidate available')
  selected=min(pool,key=lambda r:r['makespan_s']);dest=out/'selected';dest.mkdir()
  for name in ('config.yaml','target.stl','trajectory.csv'):shutil.copy2(Path(selected['job'])/name,dest/name)
  # Cached evidence is insufficient for the final deliverable: validate exact selected files again.
  final=run_validation(dest,out/'selected_validation');report.update(status=final.status,selected=selected,makespan_s=final.schedule.makespan_s,shape=final.summary_dict()['shape'],output_sha256={n:sha(dest/n) for n in ('target.stl','config.yaml','trajectory.csv')})
  if final.status!='PASS':raise RuntimeError('Selected trajectory failed full revalidation')
  report['reduction_percent']=100*(1-report['makespan_s']/report['incumbent_s']) if report['incumbent_s'] else None
 except Exception as e:report.update(status='ERROR',error=f'{type(e).__name__}: {e}')
 finally:
  report.update(elapsed_s=time.perf_counter()-started,source_changed_during_run=initial!=sources(),python=sys.version,packages={n:importlib.metadata.version(n) for n in ('numpy','shapely','polars','trimesh')});write(out/'run.json',report)
 print(json.dumps({k:report.get(k) for k in ('status','makespan_s','incumbent_s','reduction_percent','error')},ensure_ascii=False),flush=True)
 return 0 if report.get('status')=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
