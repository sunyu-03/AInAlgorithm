import bootstrap
import unittest
import numpy as np
from types import SimpleNamespace as NS
from minmin_initial import assign
class TestMinMin(unittest.TestCase):
 def env(self):
  return NS(homes=np.array([[0.,0.,0.],[10.,0.,0.],[100.,0.,0.]]),bases=np.array([[0.,0.,0.],[10.,0.,0.],[100.,0.,0.]]),robots=[NS(xy_reach_radius_mm=6.)]*3,reach_margin_mm=1.,tasks=[NS(start=(0.,0.,1.),end=(1.,0.,1.)),NS(start=(10.,0.,1.),end=(11.,0.,1.))],config=NS(process=NS(deposition_speed_mm_s=1.,travel_speed_mm_s=10.)))
 def test_reassigns_to_reachable_robot_preserving_orientation(self):
  env=self.env(); result=assign(env,{2:[(0,False),(1,True)]})
  self.assertEqual(result,{0:[(0,False)],1:[(1,True)]})
  self.assertEqual(result,assign(env,{2:[(1,True),(0,False)]}))
 def test_unreachable_fails(self):
  env=self.env();env.tasks[0]=NS(start=(50.,0.,1.),end=(51.,0.,1.))
  with self.assertRaises(ValueError):assign(env,{0:[(0,False)]})
if __name__=='__main__':unittest.main()
