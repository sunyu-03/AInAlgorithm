import sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parent
if not (ROOT/'bootstrap.py').exists():ROOT=Path(r'C:\Users\User\Desktop\AInAlgorithm\1014')
sys.path.insert(0,str(ROOT))
import bootstrap
import numpy as np
from bundle_reassign import insert_bundle,travel,bundle_candidates
from finite_stay import propose,propose_advance,compress_global_idle,schedule_finite_stay,all_parts
from resident_reassign import full_thresholds
from run_parallel import deposition

class Tests(unittest.TestCase):
 def env(self):
  h=np.array([[-10,0,0],[100,0,0],[200,0,0]],float)
  return NS(homes=h,bases=h,reach_margin_mm=1.,robots=[NS(tcp_radius_mm=.1,arm_envelope_radius_mm=.1,xy_reach_radius_mm=500.) for _ in range(3)],config=NS(collision=NS(check_tcp_radius=True,check_arm_envelope=False,geometry_epsilon_mm=1e-6,arm_clearance_mm=0.),process=NS(travel_speed_mm_s=10.,deposition_speed_mm_s=1.)),tasks=[NS(start=(float(i*3),0,1),end=(float(i*3+1),0,1)) for i in range(6)])
 def test_bundle_insertion_cost_order_and_direction(self):
  e=self.env();ss=[(0,False),(5,True)];bundle=[(1,True),(2,False),(3,True)];got,cost=insert_bundle(e,ss,bundle,1)
  self.assertAlmostEqual(travel(e,got,1)-travel(e,ss,1),cost)
  k=got.index(bundle[0]);self.assertEqual(got[k:k+3],bundle)
 def test_batch_sizes_and_multiset(self):
  e=self.env();seq={0:[(i,i%2==1) for i in range(6)]};got=bundle_candidates(e,seq,{}, {(1,0):100},1)
  self.assertEqual([c['bundle_size'] for g,c in got],[2,4])
  for g,c in got:self.assertEqual(sorted(v for ss in g.values() for v in ss),seq[0])
 def test_unreachable_bundle_excluded(self):
  e=self.env();e.robots[1].xy_reach_radius_mm=1.;e.robots[2].xy_reach_radius_mm=1.
  self.assertEqual(bundle_candidates(e,{0:[(i,False) for i in range(6)]},{},{(1,0):100},1),[])
 def test_collision_outside_wait_window_allowed(self):
  e=self.env();p=[0.,0.,1.];q=[0.,1.,3.]
  other=[[0.,100,0,0,'W'],[20.,100,0,0,'T'],[21.,*p,'T'],[22.,100,0,0,'W'],[30.,100,0,0,'W']]
  rows=[[[0.,*e.homes[0],'W'],[30.,*e.homes[0],'W']],other,[[0.,*e.homes[2],'W'],[30.,*e.homes[2],'W']]]
  own,record=propose(e,0,[0.,*p,'W'],[10.,*q,'D'],rows,full_thresholds(e),1.)
  self.assertIsNotNone(own);self.assertLess(record['arrive_s'],10.)
  # The other robot really visits the parked position, just after this window.
  self.assertIn([21.,*p,'T'],other)
 def test_unsafe_finite_wait_and_travel_rejected(self):
  e=self.env();p=[0.,0.,1.];q=[0.,1.,3.]
  rows=[[],[[0.,*p,'W'],[20.,*p,'W']],[[0.,*e.homes[2],'W'],[20.,*e.homes[2],'W']]]
  own,_=propose(e,0,[0.,*p,'W'],[10.,*q,'D'],rows,full_thresholds(e),1.);self.assertIsNone(own)
 def test_arm_wait_collision_rejected_when_tcp_is_clear(self):
  e=self.env();e.config.collision.check_arm_envelope=True;e.bases=e.bases.copy();e.bases[1]=[0.,-10.,0.];p=[0.,0.,1.];q=[1.,0.,3.]
  rows=[[],[[0.,0.,10.,1.,'W'],[20.,0.,10.,1.,'W']],[[0.,*e.homes[2],'W'],[20.,*e.homes[2],'W']]]
  self.assertIsNone(propose(e,0,[0.,*p,'W'],[10.,*q,'D'],rows,full_thresholds(e),1.)[0])
  e.config.collision.check_arm_envelope=False
  self.assertIsNotNone(propose(e,0,[0.,*p,'W'],[10.,*q,'D'],rows,full_thresholds(e),1.)[0])
 def fixture(self):
  e=self.env();e.tasks=[NS(start=(0.,0.,1.),end=(1.,0.,1.)),NS(start=(1.,0.,3.),end=(2.,0.,3.))]
  rr=[[0.,*e.homes[0],'T'],[1.,0.,0.,1.,'D'],[2.,1.,0.,1.,'T'],[3.,*e.homes[0],'W'],[8.,*e.homes[0],'T'],[9.,1.,0.,3.,'D'],[10.,2.,0.,3.,'T'],[11.,*e.homes[0],'W']]
  return e,[rr]+[[[0.,*e.homes[r],'W'],[11.,*e.homes[r],'W']] for r in (1,2)]
 def test_shortcut_reduces_global_span_preserves_beads_and_D_duration(self):
  e,rr=self.fixture();out=schedule_finite_stay(e,rr,1.);plain,info=compress_global_idle(rr)
  self.assertLess(out['schedule']['makespan_s'],info['makespan_s']);self.assertEqual(deposition(out['rows']),deposition(rr));self.assertTrue(any(v['accepted'] for v in out['schedule']['shortcuts']))
  self.assertEqual([round(b[0]-a[0],8) for a,b in zip(out['rows'][0],out['rows'][0][1:]) if a[4]=='D'],[1.,1.])
 def test_advanced_layer_obeys_other_robot_deposition_barrier(self):
  e,rr=self.fixture();got=propose_advance(e,0,rr[0][2],rr[0][5:8],6.,rr,full_thresholds(e),1.)
  self.assertIsNotNone(got);own,record=got
  firstD=next(v[0] for v in own if v[4]=='D');self.assertAlmostEqual(firstD,6.);self.assertGreater(record['advance_s'],0.)
  self.assertIsNone(propose_advance(e,0,rr[0][2],rr[0][5:8],10.,rr,full_thresholds(e),1.))
 def test_compression_never_removes_other_robot_motion(self):
  e,rr=self.fixture();rr[1]=[[0.,100,0,0,'W'],[4.,100,0,0,'T'],[7.,110,0,0,'W'],[11.,110,0,0,'W']]
  out,info=compress_global_idle(rr);self.assertAlmostEqual(info['removed_global_idle_s'],2.)
  a,b=next((a,b) for a,b in zip(out[1],out[1][1:]) if a[4]=='T');self.assertAlmostEqual(b[0]-a[0],3.)
 def test_home_free_resident_route_preserved(self):
  e,rr=self.fixture();rr[0]=[v.copy() for v in rr[0] if v[0] not in (3.,8.)];rr[0][2][4]='T'
  result=schedule_finite_stay(e,rr,1.);self.assertEqual(result['schedule']['shortcuts'],[]);self.assertEqual(deposition(result['rows']),deposition(rr))
if __name__=='__main__':unittest.main()
