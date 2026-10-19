import bootstrap
import unittest
from collections import Counter
import numpy as np
from fixed_boundary_lns import order_tasks,undirected,directed
class DirectionTests(unittest.TestCase):
 def test_single_segment_reversal_reduces_fixed_endpoint_travel(self):
  t=dict(p=np.array([0.,0,0]),q=np.array([9.,0,0]),task=0)
  result=order_tasks([t],np.array([10.,0,0]),np.array([-1.,0,0]));self.assertTrue(np.array_equal(result[0]['p'],t['q']));self.assertTrue(np.array_equal(result[0]['q'],t['p']))
 def test_order_and_direction_keep_every_geometric_task(self):
  tasks=[dict(p=np.array([a,0.,0]),q=np.array([a+1.,0,0]),task=i) for i,a in enumerate([0.,10.,2.])];start=np.array([-1.,0,0]);end=np.array([12.,0,0]);result=order_tasks(tasks,start,end)
  def cost(seq):return float(np.linalg.norm(seq[0]['p']-start)+sum(np.linalg.norm(a['q']-b['p']) for a,b in zip(seq,seq[1:]))+np.linalg.norm(seq[-1]['q']-end))
  self.assertEqual(sorted(t['task'] for t in result),[0,1,2]);self.assertLessEqual(cost(result),cost(tasks)+1e-8)
  self.assertEqual(Counter(tuple(sorted((tuple(t['p']),tuple(t['q'])))) for t in result),Counter(tuple(sorted((tuple(t['p']),tuple(t['q'])))) for t in tasks))
 def test_direction_invariant_is_explicitly_distinct(self):
  a=[[[0.,0,0,0,'D'],[1.,1,0,0,'W']]];b=[[[0.,1,0,0,'D'],[1.,0,0,0,'W']]]
  self.assertEqual(undirected(a),undirected(b));self.assertNotEqual(directed(a),directed(b))
if __name__=='__main__':unittest.main()
