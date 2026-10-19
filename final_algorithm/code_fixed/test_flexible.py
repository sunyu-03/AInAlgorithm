from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from flexible_parallel import earliest_intervals,rebalance
from types import SimpleNamespace
import numpy as np
class Tests(unittest.TestCase):
 def test_union_chain_and_closed_endpoints(self):
  self.assertGreater(earliest_intervals([[0.,2.],[1.,4.],[3.,7.]],0.),7.)
 def test_gap_is_not_skipped(self):self.assertEqual(earliest_intervals([[0.,2.],[4.,8.]],3.),3.)
 def test_empty_intervals_and_lower_bound(self):self.assertEqual(earliest_intervals([],5.),5.)
 def test_reassignment_keeps_tasks_and_respects_reach(self):
  robots=[SimpleNamespace(xy_reach_radius_mm=100.) for _ in range(3)]
  env=SimpleNamespace(tasks=[SimpleNamespace(start=(0.,0.,1.),end=(8.,0.,1.)) for _ in range(3)],bases=np.array([[0.,0.],[0.,0.],[1000.,0.]]),robots=robots,reach_margin_mm=1.,config=SimpleNamespace(process=SimpleNamespace(deposition_speed_mm_s=8.)))
  groups,moves=rebalance(env,{0:[0,1,2]});self.assertEqual(sorted(i for g in groups.values() for i in g),[0,1,2]);self.assertNotIn(2,groups);self.assertGreater(len(moves),0)
if __name__=='__main__':unittest.main()
