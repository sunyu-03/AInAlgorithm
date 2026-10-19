import unittest
from types import SimpleNamespace
from refreshed_scores import current_scores

class ScoresTest(unittest.TestCase):
    def test_offset_and_internal_wait_are_attributed_once(self):
        rows={0:[[0,0,0,1,'T'],[2,1,0,1,'W'],[5,1,0,1,'D'],[9,2,0,1,'W'],[11,2,0,1,'D'],[16,3,0,1,'T'],[20,0,0,1,'W']]}
        ref=dict(model=SimpleNamespace(rows=rows),offsets={0:30})
        wait,finish=current_scores(SimpleNamespace(ends=(10,0,0)),ref,{0:[(4,False),(5,False)]},1)
        self.assertEqual(wait,{4:23.,5:2.})
        self.assertEqual(finish,{(1,0):46})
        # Recomputed previous-layer completion changes only initial waiting.
        wait2,_=current_scores(SimpleNamespace(ends=(25,0,0)),ref,{0:[(4,False),(5,False)]},1)
        self.assertEqual(wait2,{4:8.,5:2.})

    def test_robot_availability_violation_rejected(self):
        ref=dict(model=SimpleNamespace(rows={0:[[0,0,0,1,'D'],[1,1,0,1,'W']]}),offsets={0:2})
        with self.assertRaises(ValueError):
            current_scores(SimpleNamespace(ends=(3,0,0)),ref,{0:[(1,False)]},1)

    def test_task_mismatch_rejected(self):
        ref=dict(model=SimpleNamespace(rows={0:[[0,0,0,1,'D'],[1,1,0,1,'W']]}),offsets={0:2})
        with self.assertRaises(ValueError):
            current_scores(SimpleNamespace(ends=(0,0,0)),ref,{0:[(1,False),(2,False)]},1)

if __name__=='__main__':unittest.main()
