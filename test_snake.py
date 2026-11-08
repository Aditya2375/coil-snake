import unittest
import numpy as np
from snake import Snake,Network,evaluate
class Checks(unittest.TestCase):
    def test_seed(self):
        a,b=Snake(78),Snake(78)
        for _ in range(4):
            self.assertEqual(a.body,b.body);self.assertEqual(a.food,b.food)
            if not a.done:a.step(1);b.step(1)
    def test_wall(self):
        e=Snake();e.body=[(7,4),(6,4),(5,4)];e.direction=0
        _,r,d=e.step(0);self.assertTrue(d);self.assertEqual(r,-10)
    def test_eat(self):
        e=Snake();e.food=(5,4);e.step(0);self.assertEqual(e.score,1);self.assertEqual(len(e.body),4);self.assertNotIn(e.food,e.body)
    def test_tail_vacates(self):
        e=Snake();e.body=[(3,3),(3,4),(2,4),(2,3)];e.direction=2;e.food=(7,7)
        self.assertFalse(e.collision((2,3)));e.step(0);self.assertFalse(e.done)
    def test_grads(self):
        n=Network(8);x=np.random.default_rng(3).normal(size=(4,11));a=np.array([0,1,2,0]);y=np.zeros(4)
        gs,_=n.grads(x,a,y)
        for pi,idx in [(0,(2,3)),(1,(8,)),(2,(4,1)),(3,(2,))]:
            old=n.p[pi][idx];eps=1e-5;n.p[pi][idx]=old+eps;_,p=n.grads(x,a,y);n.p[pi][idx]=old-eps;_,m=n.grads(x,a,y);n.p[pi][idx]=old
            self.assertAlmostEqual(gs[pi][idx],(p-m)/(2*eps),places=6)
    def test_exported_measurements(self):
        import json
        from pathlib import Path
        d=json.loads(Path('model.json').read_text());n=Network();n.p=[np.array(x) for x in d['weights']]
        self.assertEqual(evaluate(n,d['evaluation']['seeds']),d['evaluation']['trained'])
    def test_final_move_food(self):
        e=Snake();e.steps=1499;e.food=(5,4)
        _,r,done=e.step(0);self.assertEqual(r,10);self.assertTrue(done);self.assertEqual(e.score,1)
    def test_invalid(self):
        with self.assertRaises(ValueError):Snake().step(4)
if __name__=='__main__':unittest.main()
