import bootstrap
import unittest,copy
from types import SimpleNamespace as NS
import numpy as np
from fixed_boundary_lns import rows_clip,patch,directed,joint_schedule,optimize
class Tests(unittest.TestCase):
 def env(self):
  tasks=[NS(start=(-2.,0.,0.),end=(2.,0.,0.))]
  return NS(bases=np.array([[-10.,0.,0.],[0.,-10.,0.],[20.,20.,0.]]),homes=np.array([[-2.,0.,0.],[0.,-2.,0.],[10.,10.,0.]]),tasks=tasks,robots=[NS(tcp_radius_mm=.1,arm_envelope_radius_mm=.1,xy_reach_radius_mm=100.)]*3,reach_margin_mm=1.,config=NS(collision=NS(check_tcp_radius=True,check_arm_envelope=False,geometry_epsilon_mm=1e-6,arm_clearance_mm=.1),process=NS(travel_speed_mm_s=1.,deposition_speed_mm_s=1.)))
 def test_clip_and_deposition_boundary(self):
  rr=[[0.,0,0,0,'T'],[2.,2,0,0,'D'],[4.,4,0,0,'W'],[6.,4,0,0,'W']]
  self.assertEqual(rows_clip(rr,1.,4.)[0],[1.,1.,0.,0.,'T'])
  with self.assertRaises(ValueError):rows_clip(rr,3.,4.)
 def test_roundoff_at_deposition_end_snaps_only_within_tolerance(self):
  rr=[[0.,0,0,0,'T'],[2.,2,0,0,'D'],[4.,4,0,0,'W'],[6.,4,0,0,'W']]
  self.assertEqual(rows_clip(rr,1.,4.-1e-12)[-1][1:],[4,0,0,'W'])
  with self.assertRaises(ValueError):rows_clip(rr,1.,4.-1e-5)
 def test_patch_preserves_segments_and_translates_suffix(self):
  rr=[[0.,0,0,0,'T'],[1.,1,0,0,'D'],[2.,2,0,0,'W'],[5.,2,0,0,'T'],[6.,3,0,0,'W']]
  old=[copy.deepcopy(rr),[[0.,10,0,0,'W'],[6.,10,0,0,'W']],[[0.,20,0,0,'W'],[6.,20,0,0,'W']]]
  local=[[[0.,1,0,0,'D'],[1.,2,0,0,'W']],[[0.,10,0,0,'W'],[1.,10,0,0,'W']],[[0.,20,0,0,'W'],[1.,20,0,0,'W']]]
  new=patch(old,local,1.,5.,1.)
  self.assertEqual(directed(new),directed(old));self.assertEqual(new[0][-1][0],3.)
  self.assertEqual(new[0][0],old[0][0]);self.assertEqual(new[0][-1][1:],old[0][-1][1:])
 def test_crossing_routes_are_staggered(self):
  env=self.env();routes=[[[0.,-2,0,0,'T'],[4.,2,0,0,'W']],[[0.,0,-2,0,'T'],[4.,0,2,0,'W']],[[0.,10,10,0,'W']]]
  thresholds={(r,q):(.21,.21) for r in range(3) for q in range(3) if r!=q}
  result,info=joint_schedule(env,routes,12.,.25,thresholds);self.assertIsNotNone(result)
  rows,details=result
  def position(rr,t):
   for a,b in zip(rr,rr[1:]):
    if a[0]<=t<=b[0]:return np.array(a[1:4])+(np.array(b[1:4])-a[1:4])*(t-a[0])/(b[0]-a[0])
   return np.array(rr[-1][1:4])
  for t in np.linspace(0,details['makespan_s'],1001):self.assertGreater(np.linalg.norm(position(rows[0],t)-position(rows[1],t)),.21)
 def test_stationary_boundary_robot_cannot_be_crossed(self):
  env=self.env();routes=[[[0.,-2,0,0,'T'],[4.,2,0,0,'W']],[[0.,0,0,0,'W']],[[0.,10,10,0,'W']]]
  thresholds={(r,q):(.21,.21) for r in range(3) for q in range(3) if r!=q}
  result,_=joint_schedule(env,routes,12.,.25,thresholds);self.assertIsNone(result)
 def test_no_repair_is_exact_identity(self):
  env=self.env();rows=[[[0.,-2,0,0,'D'],[4.,2,0,0,'W']],[[0.,0,-2,0,'W'],[4.,0,-2,0,'W']],[[0.,10,10,0,'W'],[4.,10,10,0,'W']]]
  self.assertEqual(optimize(env,copy.deepcopy(rows),budget_s=0.)['rows'],rows)
if __name__=='__main__':unittest.main()
