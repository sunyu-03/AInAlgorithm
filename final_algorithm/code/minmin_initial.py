"""Deterministic Min-Min assignment of unchanged directed deposition segments."""
import numpy as np
from bundle_reassign import endpoints
from parallel_scheduler import eligible
from resident_reassign import full_thresholds
from joint_wait_search import evaluate_all
from two_layer_lookahead import Model,Engine,State,choose_lookahead,materialize

def assign(env,seq):
    pending=sorted([item for items in seq.values() for item in items])
    allowed={item: [r for r in range(3) if eligible(env,[item[0]],r)] for item in pending}
    if any(not v for v in allowed.values()): raise ValueError('Task has no reachable robot')
    groups={r:[] for r in range(3)};loads=[0.]*3;pos=env.homes.copy()
    while pending:
        choices=[]
        for item in pending:
            a,b=endpoints(env,item)
            duration=float(np.linalg.norm(b-a))/env.config.process.deposition_speed_mm_s
            for r in allowed[item]:
                finish=loads[r]+float(np.linalg.norm(a-pos[r]))/env.config.process.travel_speed_mm_s+duration
                choices.append((finish,item[0],item[1],r))
        finish,i,rev,r=min(choices);item=(i,rev)
        groups[r].append(item);loads[r]=finish;pos[r]=endpoints(env,item)[1];pending.remove(item)
    assert sorted(item for values in groups.values() for item in values)==sorted(item for values in seq.values() for item in values)
    return {r:v for r,v in groups.items() if v}

def schedule_batch(env,sequences,oldrows,step=10.,beam=2):
    thresholds=full_thresholds(env);engine=Engine(env,thresholds);state=State();history=[];details=[];models={};zs=sorted(sequences)
    def model(index):
        if index not in models:
            z=zs[index];groups=assign(env,sequences[z]);plan=evaluate_all(env,groups,step)
            models[index]=Model(env,(index,0),z,plan['rows'],dict(kind='minmin_initial',assignment={str(r):v for r,v in groups.items()}),step,thresholds)
        return models[index]
    for index,z in enumerate(zs):
        chosen,trials=choose_lookahead(engine,state,[model(index)],model(index+1) if index+1<len(zs) else None,beam)
        state=chosen['state'];history.append(dict(rows=chosen['model'].rows,offsets=dict(chosen['offsets'])))
        details.append(dict(z=z,makespan_s=state.makespan,change=chosen['model'].change))
        print(f'MinMin layer {z}: {state.makespan:.3f}s',flush=True)
        engine.forget_before(index)
        for k in list(models):
            if k<index:del models[k]
    return dict(rows=materialize(env,history,state),schedule=dict(kind='minmin_initial_two_layer',layers=details,makespan_s=state.makespan,proof_step_mm=step,priority_beam=beam))
