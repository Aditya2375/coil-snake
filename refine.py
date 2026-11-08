"""DAgger: collect teacher labels on the learner's own training states. No hidden rescue."""
import json,time
from pathlib import Path
import numpy as np
from upgrade import BoardPolicy,features,teacher,evaluation
from snake import Snake,Network,evaluate
from report import analysis

def main():
 d=np.load('imitation-data.npz');base_x=d['x'];base_y=d['y'];models=[];start=time.time()
 for seed in [121,122,123]:
  g=np.random.default_rng(seed);net=BoardPolicy(seed);x=base_x.copy();y=base_y.copy();hist=[]
  for round in range(6):
   losses=[]
   for step in range(3500):
    ids=g.integers(len(x),size=256);losses.append(net.train(x[ids],y[ids],.0006))
   ev=evaluation(lambda env:int(np.argmax(net.forward(features(env)))),range(31000,31040));hist.append(dict(round=round+1,positions=len(x),loss_mean=float(np.mean(losses)),development_mean=ev['mean']));print(seed,hist[-1], 'seconds',time.time()-start,flush=True)
   if round<5:
    nx=[];ny=[]
    for game in range(140):
     env=Snake(600000+seed*10000+round*1000+game)
     while not env.done:
      f=features(env);nx.append(f);label=teacher(env);ny.append(label);a=label if g.random()<.15 else int(np.argmax(net.forward(f)));env.step(a)
    x=np.concatenate([x,np.stack(nx)])[-100000:];y=np.concatenate([y,np.array(ny)])[-100000:]
  dev=evaluation(lambda env:int(np.argmax(net.forward(features(env)))),range(32000,32200));models.append(dict(seed=seed,weights=[p.tolist() for p in net.p],history=hist,development=dev));Path('board-candidates.json').write_text(json.dumps(models,separators=(',',':')))
 best=max(models,key=lambda m:m['development']['mean']);net=BoardPolicy(best['seed']);net.p=[np.array(p,np.float32) for p in best['weights']];testseeds=list(range(40000,41000));test=evaluation(lambda env:int(np.argmax(net.forward(features(env)))),testseeds);old=json.loads(Path('model.json').read_text());baseline=Network();baseline.p=[np.array(p) for p in old['weights']];v1=evaluate(baseline,testseeds);teach=evaluation(teacher,testseeds);fixtures=[]
 for seed in [40000,40001,40002]:
  env=Snake(seed);f=features(env);fixtures.append(dict(seed=seed,input=f.tolist(),logits=net.forward(f).tolist()))
 payload=dict(version=2,algorithm='NumPy behavior cloning + DAgger',shape=[203,128,64,3],weights=best['weights'],seed=best['seed'],training=dict(initial_games=600,initial_exploration=.04,dagger_rounds=6,updates_per_round=3500,new_games_per_round=140,teacher_action_probability=.15,batch=256,lr=.0006,training_seed_ranges='initial500000-500599; DAgger600000+seed*10000+round*1000+game',candidates=models),evaluation=dict(seeds=testseeds,board=test,v1=v1,teacher=teach),fixtures=fixtures)
 payload['analysis']=analysis(payload['evaluation']);Path('1-board-policy.json').write_text(json.dumps(payload,separators=(',',':')));print('FINAL',best['seed'],test['mean'],v1['mean'],teach['mean'],flush=True)
if __name__=='__main__':main()
