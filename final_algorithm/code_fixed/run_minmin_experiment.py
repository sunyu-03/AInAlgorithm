import bootstrap
import argparse,json,time,shutil,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from assignment_env import Task
from waam_validator import run_validation,load_config
from joint_wait_search import read_baseline
from minmin_initial import schedule_batch as minmin

from finite_stay import schedule_finite_stay
from run_bundle_stay import directed,export
from run_parallel import sha,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    baseline=a.baseline.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    inputs={n:sha(baseline/n) for n in ['target.stl','config.yaml','trajectory.csv']}
    report=dict(status='RUNNING',input_sha256=inputs,source_sha256=sources,candidates=[],errors=[])
    def persist():write(out/'run.json',dict(report,elapsed_s=time.perf_counter()-start))
    persist()
    try:
        cfg=load_config(baseline/'config.yaml');robots=sorted(cfg.robots,key=lambda r:r.id)
        raw,seq,oldrows,_=read_baseline(baseline)
        env=SimpleNamespace(config=cfg,robots=robots,homes=np.array([r.home_xyz_mm if r.home_xyz_mm is not None else r.base_xyz_mm for r in robots],float),bases=np.array([r.base_xyz_mm for r in robots],float),tasks=[Task(tuple(v['start_xyz_mm']),tuple(v['end_xyz_mm'])) for v in raw],reach_margin_mm=1.)
        expected=directed([oldrows[r] for r in range(3)])
        v=run_validation(baseline,out/'baseline_validation');report['baseline_s']=v.schedule.makespan_s
        assert v.status=='PASS','Frozen baseline failed revalidation'
        choices=[(v.schedule.makespan_s,baseline,'preserved_frozen')]
        persist()
        for name,fn in [('minmin_initial',minmin)]:
            try:
                stamp=time.perf_counter();plan=fn(env,seq,oldrows,10.,2)
                report[name+'_search_s']=time.perf_counter()-stamp
                # Both arms start at the same frozen input; postprocessing is identical.
                for suffix in ['bundle','finite_stay']:
                    stamp=time.perf_counter()
                    if suffix=='finite_stay':plan=schedule_finite_stay(env,plan['rows'],10.)
                    assert directed(plan['rows'])==expected,'Deposition geometry/direction changed'
                    label=name+'_'+suffix;folder=out/label;folder.mkdir()
                    export(folder/'job',baseline,plan['rows']);write(folder/'schedule.json',plan['schedule'])
                    value=run_validation(folder/'job',folder/'validation')
                    record=dict(policy=label,status=value.status,makespan_s=value.schedule.makespan_s,postprocess_validation_s=time.perf_counter()-stamp,trajectory_sha256=sha(folder/'job/trajectory.csv'))
                    report['candidates'].append(record)
                    if value.status=='PASS' and value.schedule.makespan_s<report['baseline_s']-1e-6:choices.append((value.schedule.makespan_s,folder/'job',label))
                    persist();print(json.dumps(record),flush=True)
            except Exception as exc:
                report['errors'].append(dict(arm=name,error=str(exc),traceback=traceback.format_exc()));persist();print(traceback.format_exc(),flush=True)
        end,job,label=min(choices,key=lambda x:x[0]);dest=out/'selected';dest.mkdir()
        for n in inputs:shutil.copy2(job/n,dest/n)
        assert inputs=={n:sha(baseline/n) for n in inputs}
        assert all(sha(dest/n)==inputs[n] for n in ['target.stl','config.yaml'])
        report.update(status='PASS',selected_s=end,selected_kind=label,reduction_percent=100*(1-end/report['baseline_s']),output_sha256={n:sha(dest/n) for n in inputs})
    except Exception as exc:
        report.update(status='ERROR',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        report['source_changed']=sources!={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')};persist()
    print(json.dumps({k:report[k] for k in ['status','baseline_s','selected_s','selected_kind','reduction_percent','errors']}),flush=True)
if __name__=='__main__':main()

