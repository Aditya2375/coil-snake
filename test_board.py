"""Board encoding, teacher legality, backprop, frozen-model integrity and split checks."""
import json,unittest
from pathlib import Path
import numpy as np
from snake import Snake
from upgrade import BoardPolicy,features,teacher
from report import analysis

class BoardTests(unittest.TestCase):
 def test_features(self):
  e=Snake(40000);f=features(e);self.assertEqual(f.shape,(203,));self.assertEqual(f[11:75].sum(),1);self.assertEqual(f[75:139].sum(),1);self.assertAlmostEqual(f[139:].sum(),6/64);np.testing.assert_array_equal(f[:11],e.state())
 def test_teacher_legal_when_available(self):
  for seed in range(10):
   e=Snake(seed)
   while not e.done:
    a=teacher(e)
    if not all(e.state()[:3]):self.assertEqual(e.state()[a],0)
    e.step(a)
 def test_cross_entropy_gradient(self):
  n=BoardPolicy(71);g=np.random.default_rng(91);x=g.normal(size=(5,203)).astype(np.float32);y=np.array([0,1,2,0,1]);n.train(x,y,lr=0)
  def loss():
   z=n.forward(x);z-=z.max(1,keepdims=True);return float((-z[np.arange(5),y]+np.log(np.exp(z).sum(1))).mean())
  for layer,index in [(0,(7,2)),(0,(80,60)),(1,(12,)),(2,(30,14)),(3,(18,)),(4,(9,2)),(5,(1,))]:
   p=n.p[layer];v=p[index].copy();eps=.003;p[index]=v+eps;plus=loss();p[index]=v-eps;minus=loss();p[index]=v
   self.assertAlmostEqual((plus-minus)/(2*eps),float(n.m[layer][index]/.1),delta=.003)
 def test_saved_policy_and_statistics(self):
  m=json.loads(Path('1-board-policy.json').read_text());n=BoardPolicy(0);n.p=[np.array(w,np.float32) for w in m['weights']];self.assertEqual(sum(p.size for p in n.p),34563)
  for f in m['fixtures']:np.testing.assert_allclose(n.forward(features(Snake(f['seed']))),f['logits'],atol=1e-5)
  self.assertEqual(m['evaluation']['seeds'],list(range(40000,41000)));self.assertEqual(m['seed'],max(m['training']['candidates'],key=lambda c:c['development']['mean'])['seed']);self.assertEqual(analysis(m['evaluation']),m['analysis'])

if __name__=='__main__':unittest.main()
