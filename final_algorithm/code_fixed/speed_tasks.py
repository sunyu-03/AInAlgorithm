"""Minimize deposition length with official whole-job shape budget.

No surface-distance restrictions or model-number-specific branches.
Layer candidates pass the configured layer/coverage/overfill limits. The
aggregate shape budget controls overall IoU; final CSV validation is mandatory.
"""
from pathlib import Path
import numpy as np
from algorithm.preprocessing.task_generator import DepositionTask,hatch_edges_parametric,choose_edges,merge_collinear
from algorithm.preprocessing.reach import split_by_reach
from algorithm.preprocessing.surface_generator import envelope
from waam_validator.config.loader import load_config
from waam_validator.shape.target import load_target_mesh,determine_evaluation_layers,slice_target_layers

def metrics(poly,edges,w,res):
 dep=envelope(edges,w,res);i=poly.intersection(dep).area;u=poly.union(dep).area
 return dict(target=poly.area,intersection=i,union=u,overfill=dep.difference(poly).area,
             iou=i/u if u else 1.,coverage=i/poly.area,over=dep.difference(poly).area/poly.area,
             length=sum(np.linalg.norm(np.asarray(b)-a) for a,b in edges))

def totals(options,chosen):
 return {k:sum(o[j]['m'][k] for o,j in zip(options,chosen)) for k in ('target','intersection','union','overfill')}

def acceptable(t,sc):
 return (t['intersection']/t['union']>=sc.minimum_overall_iou+1e-5 and
         t['intersection']/t['target']>=sc.minimum_overall_coverage+1e-5 and
         t['overfill']/t['target']<=sc.maximum_overall_overfill_ratio-1e-5)

def choose_budget(options,sc):
 # Begin from maximum IoU. Then spend actual whole-job area headroom on
 # reductions in deposited length. All official layers use the same height.
 chosen=[max(range(len(o)),key=lambda j:(o[j]['m']['iou'],-o[j]['m']['length'])) for o in options]
 if not acceptable(totals(options,chosen),sc):raise ValueError('No candidate combination meets whole-job shape limits')
 moves=[]
 for layer,o in enumerate(options):
  old=o[chosen[layer]]['m']
  for j,c in enumerate(o):
   saving=old['length']-c['m']['length']
   if saving>1e-9:
    deficit=max(1e-9,(old['intersection']-sc.minimum_overall_iou*old['union'])-(c['m']['intersection']-sc.minimum_overall_iou*c['m']['union']))
    moves.append((-saving/deficit,layer,j))
 for _,layer,j in sorted(moves):
  if options[layer][j]['m']['length']>=options[layer][chosen[layer]]['m']['length']:continue
  trial=chosen.copy();trial[layer]=j
  if acceptable(totals(options,trial),sc):chosen=trial
 return chosen,totals(options,chosen)

def generate(job):
 c=load_config(job/'config.yaml');sc=c.shape_validation;w=c.process.bead_width_mm
 mesh=load_target_mesh(job/'target.stl',c);polys=slice_target_layers(mesh,determine_evaluation_layers(mesh,{},c),c)
 options=[];layers=[];cache={}
 for layer,poly in sorted(polys.items()):
  if poly.is_empty:continue
  key=(layer%2,poly.wkb)
  if key not in cache:
   pool=[]
   for spacing in (1.15,1.10,1.05,1.0,.95,.90,.80):
    edges=hatch_edges_parametric(poly,w,layer,spacing,.5)
    if not edges:continue
    edges=split_by_reach(merge_collinear(edges),c.robots,1.)
    m=metrics(poly,edges,w,sc.polygon_buffer_resolution)
    if m['iou']>=sc.minimum_layer_iou+1e-5 and m['coverage']>=sc.minimum_overall_coverage+1e-5 and m['over']<=sc.maximum_overall_overfill_ratio-1e-5:
     pool.append(dict(edges=edges,m=m,strategy=f'hatch({spacing}w,0.5w)'))
   if not pool or max(p['m']['iou'] for p in pool)<sc.minimum_overall_iou+1e-5:
    edges,name,_=choose_edges(poly,w,layer,sc.model_dump())
    edges=split_by_reach(merge_collinear(edges),c.robots,1.)
    m=metrics(poly,edges,w,sc.polygon_buffer_resolution)
    if m['iou']>=sc.minimum_layer_iou and m['coverage']>=sc.minimum_overall_coverage and m['over']<=sc.maximum_overall_overfill_ratio:
     pool.append(dict(edges=edges,m=m,strategy='fallback:'+name))
   if not pool:raise ValueError(f'No layer candidate at {layer}')
   cache[key]=pool
  layers.append(layer);options.append(cache[key])
  print(f'Geometry layer {layer}: {len(cache[key])} feasible candidates',flush=True)
 chosen,aggregate=choose_budget(options,sc);records=[];report=[]
 for layer,pool,j in zip(layers,options,chosen):
  v=pool[j];z=c.process.build_plane_z_mm+(layer+(.5 if c.process.tcp_z_reference=='center' else 1))*c.process.layer_height_mm
  for a,b in v['edges']:records.append(DepositionTask(len(records),layer,np.array([*a,z]),np.array([*b,z])))
  report.append(dict(layer=layer,strategy=v['strategy'],**v['m']))
 return records,dict(layers=report,aggregate=aggregate,overall_iou=aggregate['intersection']/aggregate['union'])
