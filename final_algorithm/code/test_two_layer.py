import sys, unittest
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parent
if not (ROOT/'bootstrap.py').exists():ROOT=Path(r'C:\Users\User\Desktop\AInAlgorithm\1014')
sys.path.insert(0,str(ROOT))
import bootstrap
import numpy as np
from two_layer_lookahead import Model,Engine,State,choose_lookahead,materialize,split_episodes
from resident_reassign import full_thresholds
from task_swap_parallel import route
from run_parallel import deposition

class Tests(unittest.TestCase):
 def env(self):
  h=np.array([[25,0,0],[1000,0,0],[-5,0,0]],float)
  return NS(homes=h,bases=h,config=NS(collision=NS(check_tcp_radius=False,check_arm_envelope=False,geometry_epsilon_mm=1e-6,arm_clearance_mm=0.),process=NS(travel_speed_mm_s=10.,deposition_speed_mm_s=50.)),robots=[NS(tcp_radius_mm=.1,arm_envelope_radius_mm=.1,xy_reach_radius_mm=2000.) for _ in range(3)],tasks=[NS(start=(0,0,1),end=(50,0,1)),NS(start=(1000,0,3),end=(1050,0,3))])
 def setup_models(self):
  e=self.env();th=full_thresholds(e)
  def model(uid,z,r,i):return Model(e,uid,z,{r:route(e,[(i,False)],r)[0]},dict(robot=r),10.,th)
  return e,Engine(e,th),model((0,0),1,0,0),model((0,1),1,2,0),model((1,0),3,1,1)
 def test_locally_slower_assignment_wins_horizon(self):
  e,eng,a,b,n=self.setup_models();aa=eng.transitions_for(State(),a)[0];bb=eng.transitions_for(State(),b)[0]
  self.assertLess(aa['state'].makespan,bb['state'].makespan)
  chosen,trials=choose_lookahead(eng,State(),[a,b],n)
  self.assertEqual(chosen['model'].uid,b.uid)
  self.assertLess(trials[1]['projected_end_s'],trials[0]['projected_end_s'])
 def test_layer_and_own_robot_barriers(self):
  e,eng,a,b,n=self.setup_models();st=State((30.,0.,0.),50.)
  v=eng.transitions_for(st,a)[0];off=v['offsets'][0]
  self.assertGreaterEqual(off,30.);self.assertGreaterEqual(off+a.first[0],50.)
 def test_materialization_preserves_directed_geometry_and_waits(self):
  e,eng,a,b,n=self.setup_models();one=eng.transitions_for(State(),a)[0];two=eng.transitions_for(one['state'],n)[0]
  rows=materialize(e,[dict(rows=x['model'].rows,offsets=x['offsets']) for x in (one,two)],two['state'])
  self.assertEqual(deposition(rows),deposition([a.rows[0],n.rows[1]]))
  for rr in rows:self.assertTrue(all(y[0]>x[0] for x,y in zip(rr,rr[1:])))
  seq={1:{0:[(0,False)]},3:{1:[(1,False)]}}
  episodes,resident=split_episodes(e,seq,dict(enumerate(rows)))
  self.assertFalse(resident);self.assertEqual(set(episodes),{1,3})
 def test_stationary_work_wait_reservation(self):
  e=self.env();e.config.collision.check_tcp_radius=True
  e.homes=np.array([[-10,0,0],[0,-10,0],[100,100,0]],float);e.bases=e.homes
  th=full_thresholds(e);eng=Engine(e,th)
  ar=[[0,-10,0,0,'T'],[1,0,0,1,'W'],[10,0,0,1,'D'],[11,1,0,1,'T'],[12,-10,0,0,'W']]
  br=[[0,0,-10,0,'T'],[1,0,0,3,'D'],[2,0,1,3,'T'],[3,0,-10,0,'W']]
  a=Model(e,(0,0),1,{0:ar},{},1.,th);b=Model(e,(1,0),3,{1:br},{},1.,th)
  st=eng.transitions_for(State(),a)[0]['state'];v=eng.transitions_for(st,b)[0]
  self.assertGreater(v['offsets'][1],10.)
 def test_parked_home_conflict_rejected(self):
  e=self.env();e.config.collision.check_tcp_radius=True;e.homes[1]=[0,0,0]
  with self.assertRaisesRegex(ValueError,'Parked-home'):Model(e,(0,0),1,{0:route(e,[(0,False)],0)[0]},{},1.,full_thresholds(e))
 def test_expired_part_pruning_matches_full_geometry(self):
  e,eng,a,b,n=self.setup_models();e.config.collision.check_tcp_radius=True
  # Rebuild with certified, separated homes and intersecting active paths.
  e.homes=np.array([[-10,0,0],[0,-10,0],[100,100,0]],float);e.bases=e.homes
  th=full_thresholds(e)
  ar=[[0,-10,0,0,'T'],[1,0,0,1,'W'],[10,0,0,1,'D'],[11,1,0,1,'T'],[12,-10,0,0,'W']]
  br=[[0,0,-10,0,'T'],[1,0,0,3,'D'],[2,0,1,3,'T'],[3,0,-10,0,'W']]
  a=Model(e,(0,0),1,{0:ar},{},1.,th);b=Model(e,(1,0),3,{1:br},{},1.,th)
  class Full(Engine):
   def pairs(self,a,r,b,q,cutoff=None):return super().pairs(a,r,b,q)
  self.assertEqual(Engine(e,th).pairs(b,1,a,0,100.).shape,(0,2))
  for lower in (0.,10.,11.,12.,20.):
   pruned=Engine(e,th);full=Full(e,th);state=pruned.transitions_for(State(),a)[0]['state'];state=State(state.ends,max(state.barrier,lower+1),state.reservations)
   x=pruned.transitions_for(state,b)[0];y=full.transitions_for(state,b)[0]
   self.assertEqual(x['offsets'],y['offsets'])
if __name__=='__main__':unittest.main()
