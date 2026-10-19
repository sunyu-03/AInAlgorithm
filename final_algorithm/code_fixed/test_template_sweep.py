import bootstrap
import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from template_sweep import optimize,template_groups,rules_from_seed
from fixed_boundary_lns import directed
class SweepTests(unittest.TestCase):
 def env(self):
  return NS(bases=np.array([[-10.,0,0],[0,-10.,0],[20.,20.,0]]),homes=np.array([[0.,0,0],[0,20.,0],[20.,20.,0]]),tasks=[NS(start=(0.,0,0),end=(2.,0,0))],robots=[NS(tcp_radius_mm=.1,arm_envelope_radius_mm=.1,xy_reach_radius_mm=100.)]*3,reach_margin_mm=1.,config=NS(collision=NS(check_tcp_radius=True,check_arm_envelope=False,geometry_epsilon_mm=1e-6,arm_clearance_mm=.1),process=NS(travel_speed_mm_s=1.,deposition_speed_mm_s=1.)))
 def rule(self):return dict(kind='spatial_joint',pair=[0,1],bundle_size=10,quantile=.5,flip=False,axis=[1.,0.])
 def test_sweep_maintains_future_times_after_multiple_repairs(self):
  rows=[[[0.,0,0,0,'W'],[2.,0,0,0,'D'],[5.,2,0,0,'T'],[7.,0,0,1,'D'],[10.,2,0,1,'W'],[12.,2,0,1,'W']],[[0.,0,20,0,'W'],[12.,0,20,0,'W']],[[0.,20,20,0,'W'],[12.,20,20,0,'W']]]
  with tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parent.parent)) as tmp:
   assert Path(tmp).resolve().is_relative_to(Path(__file__).resolve().parent.parent)
   seed=Path(tmp)/'seed.json';seed.write_text(json.dumps(dict(layers=[dict(accepted=True,saved_s=1,z=0.,change=self.rule())])))
   plan=optimize(self.env(),rows,budget_s=60.,seed_schedule=seed)
  self.assertEqual(directed(plan['rows']),directed(rows));self.assertEqual(plan['schedule']['accepted_count'],2);self.assertAlmostEqual(plan['schedule']['saved_s'],2.);self.assertAlmostEqual(plan['rows'][0][-1][0],10.)
 def test_selected_task_not_duplicated_when_reassigned(self):
  env=self.env();items=[dict(robot=1,start=1.,end=2.,p=np.array([0.,0,0]),q=np.array([2.,0,0]))]
  groups=template_groups(env,items,self.rule());self.assertEqual(sum(map(len,groups.values())),1);self.assertIs(groups[0][0],items[0])
if __name__=='__main__':unittest.main()

