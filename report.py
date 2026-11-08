"""Reproducible paired summary. This does not choose or train a model."""
import numpy as np

def analysis(evaluation):
    board=np.array([r['score'] for r in evaluation['board']['results']])
    old=np.array(evaluation['v1']['scores']);teacher=np.array([r['score'] for r in evaluation['teacher']['results']]);delta=board-old
    rng=np.random.default_rng(172)
    boot=np.array([rng.choice(delta,size=len(delta),replace=True).mean() for _ in range(10000)])
    return dict(paired_mean_difference=float(delta.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),bootstrap_seed=172,bootstrap_resamples=10000,better=int((delta>0).sum()),same=int((delta==0).sum()),worse=int((delta<0).sum()),board_std=float(board.std(ddof=1)),v1_std=float(old.std(ddof=1)),board_filled=int((board==61).sum()),teacher_filled=int((teacher==61).sum()))
