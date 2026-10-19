"""Run the same five neighborhood stages for any feasible WAAM input job."""
from pathlib import Path
import argparse,subprocess,json,sys,time,os,shutil,hashlib

def build_archive(plans,dest):
 layers=[]
 for p in plans:
  data=json.loads(Path(p).read_text())
  layers.extend(v for v in data.get('layers',[]) if v.get('accepted') and v.get('change',{}).get('kind')=='spatial_joint')
 Path(dest).write_text(json.dumps(dict(layers=layers,provenance=[str(p) for p in plans],scope='Validated successful repairs from the same run'),indent=2))
 return len(layers)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True,help='Directory with target.stl, config.yaml and a feasible trajectory.csv');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--search-s',type=float,default=120.);ap.add_argument('--sweep-s',type=float,default=1200.);a=ap.parse_args()
 if min(a.search_s,a.sweep_s)<=0:ap.error('Time budgets must be positive')
 root=Path(__file__).resolve().parent;base=a.input.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();current=base;history=[];seeds={};best_source=None;successful_plans=[]
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 inputs={n:sha(base/n) for n in ['target.stl','config.yaml','trajectory.csv']}
 # The packaged layout has code_fixed and code; development uses sibling experiments.
 fixed=root/'code_fixed';free=root/'code'
 if not fixed.exists():fixed=root.parent/'fixed_boundary_sweep_20261019_002/code'
 if not free.exists():free=root.parent/'fixed_boundary_direction_20261019_001/code'
 for name,code,mode,budget in [('fixed_explore',fixed,'explore',a.search_s),('fixed_sweep',fixed,'sweep',a.sweep_s),('direction_explore',free,'explore',a.search_s),('direction_sweep',free,'sweep',a.sweep_s),('memory_sweep',free,'sweep',a.sweep_s)]:
  seedkey='fixed' if name.startswith('fixed') else 'direction';dest=out/name
  if name=='memory_sweep':
   archive=out/'memory_seed_archive.json'
   if not build_archive(successful_plans,archive):history.append(dict(stage=name,status='SKIPPED_NO_SUCCESSFUL_RULES'));continue
   seeds['direction']=archive
  if mode=='sweep' and seedkey not in seeds:history.append(dict(stage=name,status='SKIPPED_SEED_UNAVAILABLE'));continue
  script=code/('run_fixed_lns.py' if mode=='explore' else 'run_sweep.py');command=[sys.executable,'-B',str(script),'--baseline',str(current),'--out',str(dest),'--budget-s',str(budget)]
  if mode=='sweep':command+=['--seed-plan',str(seeds[seedkey])]
  print('START '+name,flush=True)
  with (out/(name+'.log')).open('w',encoding='utf-8') as log:
   try:
    result=subprocess.run(command,env=os.environ,stdout=log,stderr=subprocess.STDOUT,timeout=budget+600)
    r=json.loads((dest/'run.json').read_text()) if (dest/'run.json').exists() else {};record=dict(stage=name,status=r.get('status','ERROR'),exit_code=result.returncode)
    if r.get('status')=='PASS' and r.get('source_changed') is False:
     current=dest/'selected';best_source=dest;record.update(baseline_s=r['baseline_s'],selected_s=r['selected_s'],saved_s=r['baseline_s']-r['selected_s'])
     if mode=='explore':seeds[seedkey]=dest/'schedule.json'
     if r.get('candidate_status')=='PASS':successful_plans.append(dest/'schedule.json')
    else:record.update(status='ERROR',error=r.get('error','Stage did not finish successfully; previous selected input retained'))
   except subprocess.TimeoutExpired:record=dict(stage=name,status='TIMEOUT',error='Previous selected input retained')
  history.append(record);(out/'progress.json').write_text(json.dumps(history,indent=2));print(json.dumps(record),flush=True)
 if best_source is None:raise RuntimeError('No stage validated the input; no output selected')
 (out/'selected').mkdir()
 for n in inputs:shutil.copy2(current/n,out/'selected'/n)
 r=json.loads((best_source/'run.json').read_text());validation=best_source/('baseline_validation' if r['selected_kind']=='preserved_input' else 'candidate_validation');shutil.copytree(validation,out/'selected_validation')
 assert all(sha(base/n)==h for n,h in inputs.items())
 report=dict(status='PASS',input_directory=str(base),input_sha256=inputs,stages=history,selected_s=r['selected_s'],selected_source=str(best_source),selected_sha256={n:sha(out/'selected'/n) for n in inputs},elapsed_s=time.perf_counter()-start)
 (out/'pipeline_run.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()

