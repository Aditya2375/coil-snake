"""Seeded Snake environment and a small NumPy DQN. No ML framework."""
import argparse, json, random
from collections import deque
from pathlib import Path
import numpy as np

class RNG:
    """32-bit LCG shared with the browser, so seeded games match exactly."""
    def __init__(self,seed): self.value=seed & 0xffffffff
    def random(self):
        self.value=(1664525*self.value+1013904223)&0xffffffff
        return self.value/4294967296
    def choice(self,items): return items[int(self.random()*len(items))]
    def randrange(self,n): return int(self.random()*n)

class Snake:
    directions = ((1,0),(0,1),(-1,0),(0,-1))
    def __init__(self, seed=0, size=8):
        self.rng=RNG(seed); self.size=size; self.reset()
    def reset(self):
        c=self.size//2; self.body=[(c,c),(c-1,c),(c-2,c)]; self.direction=0
        self.score=0; self.steps=0; self.hunger=0; self.done=False; self.food=self.spawn(); return self.state()
    def spawn(self):
        free=[(x,y) for y in range(self.size) for x in range(self.size) if (x,y) not in self.body]
        return self.rng.choice(free) if free else None
    def collision(self,p,growing=False):
        x,y=p
        return x<0 or y<0 or x>=self.size or y>=self.size or p in (self.body if growing else self.body[:-1])
    def state(self):
        x,y=self.body[0]; fx,fy=self.food if self.food else (x,y)
        dangers=[]
        for a in (0,1,2):
            d=(self.direction+(0,1,-1)[a])%4; dx,dy=self.directions[d];p=(x+dx,y+dy)
            dangers.append(float(self.collision(p,p==self.food)))
        return np.array(dangers+[float(self.direction==d) for d in range(4)]+[float(fx<x),float(fx>x),float(fy<y),float(fy>y)],dtype=np.float64)
    def step(self,action):
        if self.done: raise ValueError('episode finished')
        if action not in (0,1,2): raise ValueError('invalid action')
        x,y=self.body[0]; old=abs(x-self.food[0])+abs(y-self.food[1]);self.direction=(self.direction+(0,1,-1)[action])%4
        dx,dy=self.directions[self.direction];p=(x+dx,y+dy);self.steps+=1;self.hunger+=1
        if self.collision(p,p==self.food):
            self.done=True; return self.state(),-10.,True
        self.body.insert(0,p); reward=-.03
        if p==self.food:
            self.score+=1;self.hunger=0;reward=10.;self.food=self.spawn()
            if self.food is None:self.done=True;return self.state(),reward,True
        else:
            self.body.pop();new=abs(p[0]-self.food[0])+abs(p[1]-self.food[1]);reward+=.08*(old-new)
        if self.hunger>=100 or self.steps>=1500:
            self.done=True
            if reward != 10.: reward=-10.
        return self.state(),reward,self.done

class Network:
    def __init__(self, seed=0):
        g=np.random.default_rng(seed)
        self.p=[g.normal(0,np.sqrt(2/11),(11,64)),np.zeros(64),g.normal(0,np.sqrt(2/64),(64,3)),np.zeros(3)]
        self.m=[np.zeros_like(p) for p in self.p];self.v=[np.zeros_like(p) for p in self.p];self.t=0
    def forward(self,x):
        a=np.maximum(0,x@self.p[0]+self.p[1]);return a@self.p[2]+self.p[3]
    def grads(self,x,actions,targets):
        z=x@self.p[0]+self.p[1];h=np.maximum(0,z);q=h@self.p[2]+self.p[3];diff=q[np.arange(len(x)),actions]-targets
        # Huber loss derivative, bounded to reduce instability.
        d=np.zeros_like(q);d[np.arange(len(x)),actions]=np.clip(diff,-1,1)/len(x)
        dh=(d@self.p[2].T)*(z>0)
        return [x.T@dh,dh.sum(0),h.T@d,d.sum(0)],float(np.mean(np.where(abs(diff)<1,.5*diff**2,abs(diff)-.5)))
    def train(self,x,a,y,lr=.001):
        grad,loss=self.grads(x,a,y);self.t+=1
        for i in range(4):
            g=np.clip(grad[i],-5,5);self.m[i]=.9*self.m[i]+.1*g;self.v[i]=.999*self.v[i]+.001*g*g
            self.p[i]-=lr*(self.m[i]/(1-.9**self.t))/(np.sqrt(self.v[i]/(1-.999**self.t))+1e-8)
        return loss
    def export(self):return [p.tolist() for p in self.p]

def evaluate(net,seeds,random_policy=False):
    scores=[];steps=[]
    for seed in seeds:
        env=Snake(seed);g=RNG(seed+90000);s=env.state()
        while not env.done:
            a=g.randrange(3) if random_policy else int(np.argmax(net.forward(s)))
            s,_,_=env.step(a)
        scores.append(env.score);steps.append(env.steps)
    return dict(scores=scores,mean=float(np.mean(scores)),median=float(np.median(scores)),best=max(scores),mean_steps=float(np.mean(steps)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--episodes',type=int,default=2400);ap.add_argument('--out',default='model.json');args=ap.parse_args()
    random.seed(731);g=np.random.default_rng(731);net=Network(731);target=Network(731);memory=deque(maxlen=30000);history=[];total=0;loss=0
    for ep in range(args.episodes):
        env=Snake(ep+100);s=env.state();epsilon=max(.04,1-ep/(args.episodes*.7))
        while not env.done:
            a=random.randrange(3) if random.random()<epsilon else int(np.argmax(net.forward(s)))
            ns,r,done=env.step(a);memory.append((s,a,r,ns,done));s=ns;total+=1
            if len(memory)>=128 and total%4==0:
                ix=g.choice(len(memory),64,replace=False);batch=[memory[i] for i in ix];x=np.array([b[0] for b in batch]);acts=np.array([b[1] for b in batch]);rewards=np.array([b[2] for b in batch]);nxt=np.array([b[3] for b in batch]);ends=np.array([b[4] for b in batch]);best=net.forward(nxt).argmax(1);y=rewards+.95*(1-ends)*target.forward(nxt)[np.arange(64),best];loss=net.train(x,acts,y)
            if total%400==0:target.p=[p.copy() for p in net.p]
        history.append(dict(episode=ep+1,score=env.score,epsilon=round(epsilon,4),loss=round(loss,5)))
        if (ep+1)%200==0:print(ep+1,round(np.mean([h['score'] for h in history[-100:]]),2),flush=True)
    seeds=list(range(20000,20100));ev=evaluate(net,seeds);baseline=evaluate(net,seeds,True)
    payload=dict(version=1,algorithm='Double DQN',shape=[11,64,3],seed=731,episodes=args.episodes,steps=total,board_size=8,weights=net.export(),history=history,evaluation=dict(seeds=seeds,trained=ev,random=baseline),state_order=['danger ahead','danger right','danger left','facing right','facing down','facing left','facing up','food left','food right','food above','food below'])
    Path(args.out).write_text(json.dumps(payload,separators=(',',':')));print('EVALUATION',ev,'RANDOM',baseline)
if __name__=='__main__':main()
