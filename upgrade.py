"""Board-aware NumPy imitation experiment. Teacher and policy are explicitly distinct."""
import argparse,json,time
from collections import deque
from pathlib import Path
import numpy as np
from snake import Snake,RNG,evaluate,Network
DIRS=Snake.directions

def reach(start,blocked,size=8):
 seen={start};q=deque([start]);distance={start:0}
 while q:
  p=q.popleft()
  for dx,dy in DIRS:
   n=(p[0]+dx,p[1]+dy)
   if 0<=n[0]<size and 0<=n[1]<size and n not in blocked and n not in seen:seen.add(n);distance[n]=distance[p]+1;q.append(n)
 return distance

def teacher(env):
 """One-step flood-fill safety, then shortest visible food distance. No future food access."""
 options=[]
 for a in range(3):
  d=(env.direction+(0,1,-1)[a])%4;dx,dy=DIRS[d];p=(env.body[0][0]+dx,env.body[0][1]+dy);eat=p==env.food
  if env.collision(p,eat):continue
  body=[p]+env.body if eat else [p]+env.body[:-1]
  dist=reach(p,set(body[1:-1]),env.size);tail=body[-1] in dist;space=len(dist);food=dist.get(env.food,999)
  options.append((a,tail,space,food,eat))
 if not options:return 0
 safe=[o for o in options if o[1] and o[2]>=min(len(env.body)+2,12)]
 if safe:return max(safe,key=lambda o:(o[4],-o[3],o[2],-o[0]))[0]
 return max(options,key=lambda o:(o[1],o[2],-o[3],-o[0]))[0]

def features(env):
 # Head location, food location, and body release order plus v1 engineered observations.
 v=np.zeros((3,8,8),np.float32)
 for i,(x,y) in enumerate(env.body):v[2,y,x]=(len(env.body)-i)/64
 x,y=env.body[0];v[0,y,x]=1
 if env.food:x,y=env.food;v[1,y,x]=1
 return np.concatenate([env.state().astype(np.float32),v.ravel()])
class BoardPolicy:
 def __init__(self,seed):
  g=np.random.default_rng(seed);sizes=[203,128,64,3];self.p=[]
  for a,b in zip(sizes,sizes[1:]):self.p.extend([g.normal(0,np.sqrt(2/a),(a,b)).astype(np.float32),np.zeros(b,np.float32)])
  self.m=[np.zeros_like(p) for p in self.p];self.v=[np.zeros_like(p) for p in self.p];self.t=0
 def forward(self,x):
  for i in range(0,6,2):x=x@self.p[i]+self.p[i+1];x=np.maximum(0,x) if i<4 else x
  return x
 def train(self,x,y,lr=.001):
  z1=x@self.p[0]+self.p[1];h1=np.maximum(0,z1);z2=h1@self.p[2]+self.p[3];h2=np.maximum(0,z2);z3=h2@self.p[4]+self.p[5];p=np.exp(z3-z3.max(1,keepdims=True));p/=p.sum(1,keepdims=True);loss=-np.log(np.maximum(p[np.arange(len(x)),y],1e-9)).mean();p[np.arange(len(x)),y]-=1;p/=len(x);d2=(p@self.p[4].T)*(z2>0);d1=(d2@self.p[2].T)*(z1>0);grad=[x.T@d1,d1.sum(0),h1.T@d2,d2.sum(0),h2.T@p,p.sum(0)];self.t+=1
  for i,d in enumerate(grad):
   self.m[i]=.9*self.m[i]+.1*d;self.v[i]=.999*self.v[i]+.001*d*d;self.p[i]-=lr*(self.m[i]/(1-.9**self.t))/(np.sqrt(self.v[i]/(1-.999**self.t))+1e-8)
  return float(loss)
def evaluation(policy,seeds):
 out=[]
 for seed in seeds:
  env=Snake(seed)
  while not env.done:env.step(policy(env))
  out.append(dict(seed=seed,score=env.score,steps=env.steps))
 return dict(mean=float(np.mean([x['score'] for x in out])),median=float(np.median([x['score'] for x in out])),results=out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--games',type=int,default=600);ap.add_argument('--updates',type=int,default=6000);args=ap.parse_args();g=np.random.default_rng(121);xs=[];ys=[];start=time.time()
 # Aggregation includes imperfect exploratory states, not only ideal teacher play.
 for game in range(args.games):
  env=Snake(500000+game)
  while not env.done:
   label=teacher(env);xs.append(features(env));ys.append(label);a=int(g.integers(3)) if g.random()<.04 else label;env.step(a)
  if (game+1)%100==0:print('data',game+1,len(xs),round(time.time()-start,1),flush=True)
 x=np.stack(xs);y=np.array(ys);np.savez_compressed('imitation-data.npz',x=x,y=y);val=BoardPolicy(121);history=[]
 for step in range(1,args.updates+1):
  ids=g.integers(len(x),size=256);loss=val.train(x[ids],y[ids])
  if step%500==0:history.append(dict(update=step,loss=loss));print('train',step,loss,round(time.time()-start,1),flush=True)
 seeds=list(range(30000,30100));ev=evaluation(lambda env:int(np.argmax(val.forward(features(env)))),seeds);teacher_ev=evaluation(teacher,seeds);old=json.loads(Path('model.json').read_text());net=Network();net.p=[np.array(v) for v in old['weights']];baseline=evaluate(net,seeds);payload=dict(version=2,experiment='NumPy board-aware policy distilled from a labeled flood-fill teacher, not reinforcement learning',shape=[203,128,64,3],weights=[p.tolist() for p in val.p],training=dict(games=args.games,positions=len(x),updates=args.updates,seed=121,history=history),evaluation=ev,teacher_evaluation=teacher_ev,v1_evaluation=baseline,fixtures=[])
 for seed in [30000,30001,30002]:
  env=Snake(seed);f=features(env);payload['fixtures'].append(dict(seed=seed,input=f.tolist(),logits=val.forward(f).tolist()))
 Path('1-board-policy.json').write_text(json.dumps(payload,separators=(',',':')));print('EVALUATION',ev['mean'],'teacher',teacher_ev['mean'],'v1',baseline['mean'],round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
