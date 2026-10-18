import sys,unittest
from types import SimpleNamespace as NS
sys.path.insert(0,r'C:\Users\User\Desktop\AInAlgorithm\1014')
import bootstrap
import numpy as np
from shapely import MultiPoint,STRtree
from resident_reassign import insertion,explicit_route,assignment_candidates,wait_scores,parked_proof,resident_layer,full_thresholds
class Tests(unittest.TestCase):
 def env(self):
  homes=np.array([[0,0,100],[100,0,100],[200,0,100]],float)
  c=NS(check_tcp_radius=True,check_arm_envelope=False,geometry_epsilon_mm=1e-6,arm_clearance_mm=0.)
  return NS(config=NS(collision=c,process=NS(travel_speed_mm_s=150.,deposition_speed_mm_s=8.)),homes=homes,bases=homes,reach_margin_mm=1.,robots=[NS(tcp_radius_mm=.1,arm_envelope_radius_mm=.1,xy_reach_radius_mm=300.) for _ in range(3)],tasks=[NS(start=(10.,0.,1.),end=(20.,0.,1.)),NS(start=(40.,0.,1.),end=(50.,0.,1.)),NS(start=(70.,0.,1.),end=(80.,0.,1.))])
 def test_insertion_cost_and_direction(self):
  env=self.env();seq=[(0,False),(2,True)];new,cost=insertion(env,seq,(1,True),0)
  def travel(s):
   rr=explicit_route(env,s,0,env.homes[0],env.homes[0]);return sum(b[0]-a[0] for a,b in zip(rr,rr[1:]) if a[4]=='T')*150
  self.assertAlmostEqual(travel(new)-travel(seq),cost);self.assertIn((1,True),new)
 def test_assignment_preserves_task_multiset_and_direction(self):
  env=self.env();seq={0:[(0,False),(1,True)],1:[(2,False)]}
  got=assignment_candidates(env,seq,{0:10.,1:0.,2:0.},{(1,0):100.,(1,1):10.},1)
  self.assertTrue(got)
  for g,change in got:self.assertEqual(sorted(v for ss in g.values() for v in ss),[(0,False),(1,True),(2,False)])
 def test_unreachable_destination_excluded(self):
  env=self.env();env.robots[1].xy_reach_radius_mm=1.;env.robots[2].xy_reach_radius_mm=1.
  self.assertEqual(assignment_candidates(env,{0:[(0,False)]},{0:10.},{(1,0):100.},1),[])
 def test_tcp_stationary_certificate(self):
  tree=STRtree([MultiPoint([(0,0)]).convex_hull]);c=NS(check_tcp_radius=True,check_arm_envelope=False)
  self.assertFalse(parked_proof((0,0,1),(100,0),(tree,tree),(1.,1.),c));self.assertTrue(parked_proof((5,0,1),(100,0),(tree,tree),(1.,1.),c))
 def test_arm_stationary_certificate(self):
  tcp=STRtree([MultiPoint([(50,5)]).convex_hull]);arm=STRtree([MultiPoint([(50,-5),(50,5)]).convex_hull]);c=NS(check_tcp_radius=False,check_arm_envelope=True)
  self.assertFalse(parked_proof((100,0,1),(0,0),(tcp,arm),(.1,.1),c))
 def test_unsafe_resident_end_rejected(self):
  env=self.env();env.tasks=[NS(start=(5.,0.,1.),end=(6.,0.,1.)),NS(start=(6.,0.,1.),end=(7.,0.,1.))];ends=env.homes.copy();ends[0]=[6,0,1]
  with self.assertRaises(ValueError):resident_layer(env,{0:[(0,False)],1:[(1,False)]},env.homes,ends,full_thresholds(env),10.)
 def test_wait_targets_next_task(self):
  rows={0:[[0,0,0,1,'W'],[5,0,0,1,'D'],[7,1,0,1,'W']]};w,f=wait_scores(rows,{1:{0:[(0,False)]}});self.assertEqual(w[0],5.);self.assertEqual(f[1,0],7.)
if __name__=='__main__':unittest.main()
