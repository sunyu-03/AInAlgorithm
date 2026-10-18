from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from run_parallel import deposition,overlap
from parallel_scheduler import parts,earliest
from types import SimpleNamespace
import numpy as np

class Tests(unittest.TestCase):
 def test_proof_subdivision_does_not_modify_csv(self):
  rows=[[0.,0.,0.,1.,'D'],[10.,100.,0.,1.,'W']];before=[r.copy() for r in rows]
  g=parts(rows,np.array([-1000.,0.]),25.)
  self.assertEqual(len(g[0]),4);self.assertEqual(rows,before);self.assertEqual(g[2].tolist(),[[0.,2.5],[2.5,5.],[5.,7.5],[7.5,10.]])
 def test_direction_independent_multiset(self):
  a=[[[0.,0.,0.,1.,'D'],[1.,8.,0.,1.,'W']]];b=[[[0.,8.,0.,1.,'D'],[1.,0.,0.,1.,'W']]]
  self.assertEqual(deposition(a),deposition(b))
 def test_actual_motion_and_deposition_overlap(self):
  rows=[[[0.,0.,0.,0.,'T'],[5.,1.,0.,0.,'W']],[[0.,0.,0.,0.,'W'],[2.,0.,0.,0.,'D'],[7.,1.,0.,0.,'W']]]
  self.assertEqual(overlap(rows),dict(simultaneous_motion_s=3.,simultaneous_deposition_s=0.))
 def test_mode_switch_keeps_robot_active(self):
  rows=[[[0.,0.,0.,0.,'T'],[2.,1.,0.,0.,'D'],[4.,2.,0.,0.,'W']],[[0.,0.,0.,0.,'T'],[4.,1.,0.,0.,'W']]]
  self.assertEqual(overlap(rows)['simultaneous_motion_s'],4.)
 def dummy_env(self):
  c=SimpleNamespace(check_tcp_radius=True,check_arm_envelope=False,geometry_epsilon_mm=1e-6,arm_clearance_mm=0.)
  return SimpleNamespace(config=SimpleNamespace(collision=c),homes=np.array([[-100.,0.,0.],[100.,0.,0.],[0.,100.,0.]]),robots=[SimpleNamespace(tcp_radius_mm=.1,arm_envelope_radius_mm=0.) for _ in range(3)],tasks=[SimpleNamespace(start=(0.,0.,0.),end=(10.,0.,0.))])
 def test_conflicting_motion_is_delayed_past_reserved_interval(self):
  env=self.dummy_env();old=parts([[5.,0.,0.,0.,'T'],[15.,10.,0.,0.,'W']],np.array([100.,0.]),10.)
  new=parts([[0.,10.,0.,0.,'T'],[10.,0.,0.,0.,'W']],np.array([-100.,0.]),10.)
  self.assertGreater(earliest(env,0,new,{1:old},0.),15.)
 def test_spatially_separated_motion_can_start_immediately(self):
  env=self.dummy_env();old=parts([[5.,0.,100.,0.,'T'],[15.,10.,100.,0.,'W']],np.array([100.,100.]),10.)
  new=parts([[0.,10.,0.,0.,'T'],[10.,0.,0.,0.,'W']],np.array([-100.,0.]),10.)
  self.assertEqual(earliest(env,0,new,{1:old},0.),0.)
if __name__=='__main__':unittest.main()
