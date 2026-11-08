import json
from pathlib import Path
import numpy as np
from upgrade import BoardPolicy,features,evaluation,teacher
from snake import Snake,Network,evaluate
from report import analysis
models=json.loads(Path('board-candidates.json').read_text())
best=max(models,key=lambda m:m['development']['mean']);net=BoardPolicy(best['seed']);net.p=[np.array(p,np.float32) for p in best['weights']];testseeds=list(range(40000,41000));test=evaluation(lambda env:int(np.argmax(net.forward(features(env)))),testseeds);old=json.loads(Path('model.json').read_text());baseline=Network();baseline.p=[np.array(p) for p in old['weights']];v1=evaluate(baseline,testseeds);teach=evaluation(teacher,testseeds);fixtures=[]
for seed in [40000,40001,40002]:
 env=Snake(seed);f=features(env);fixtures.append(dict(seed=seed,input=f.tolist(),logits=net.forward(f).tolist()))
payload=dict(version=2,algorithm='NumPy behavior cloning + DAgger',shape=[203,128,64,3],weights=best['weights'],seed=best['seed'],training=dict(initial_games=600,initial_exploration=.04,dagger_rounds=6,updates_per_round=3500,new_games_per_round=140,teacher_action_probability=.15,batch=256,lr=.0006,training_seed_ranges='initial500000-500599; DAgger600000+seed*10000+round*1000+game',candidates=models),evaluation=dict(seeds=testseeds,board=test,v1=v1,teacher=teach),fixtures=fixtures)
payload['analysis']=analysis(payload['evaluation']);Path('1-board-policy.json').write_text(json.dumps(payload,separators=(',',':')));print('FINAL',best['seed'],test['mean'],v1['mean'],teach['mean'],flush=True)
