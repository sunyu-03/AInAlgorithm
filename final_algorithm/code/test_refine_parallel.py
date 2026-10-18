from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from refine_parallel import reverse_rows,reflected_times,solve_offsets
from run_parallel import deposition
import numpy as np
class Tests(unittest.TestCase):
 def test_reversed_route_preserves_deposition_and_duration(self):
  rows=[[0.,-100.,0.,0.,'T'],[1.,0.,0.,1.,'D'],[2.,8.,0.,1.,'T'],[4.,-100.,0.,0.,'W']]
  r=reverse_rows(rows);self.assertEqual(r[0][0],0.);self.assertEqual(r[-1][0],4.);self.assertEqual(deposition([rows]),deposition([r]));self.assertEqual(reverse_rows(r),rows)
 def test_reflected_piece_time(self):
  self.assertEqual(reflected_times(np.array([[1.,3.],[4.,5.]]),10.).tolist(),[[7.,9.],[5.,6.]])
 def test_offset_solver_avoids_conflict_interval(self):
  end,order,starts=solve_offsets([0,1],{(0,1):[[-5.,5.]],(1,0):[[-5.,5.]]},{0:10.,1:10.})
  self.assertGreater(abs(starts[0]-starts[1]),5.);self.assertLess(end,20.)
 def test_disjoint_routes_can_overlap_fully(self):
  end,_,starts=solve_offsets([0,1],{(0,1):[],(1,0):[]},{0:10.,1:8.})
  self.assertEqual(end,10.);self.assertEqual(starts,{0:0.,1:0.})
if __name__=='__main__':unittest.main()
