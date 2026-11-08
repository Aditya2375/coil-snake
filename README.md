# Coil: NumPy Snake

[Open the live laboratory](https://aditya2375.github.io/coil-snake/)

A small Double DQN learned from scratch in NumPy, with a browser laboratory that runs the exported weights locally. Watch the network, inspect its observations and Q-values, compare the random baseline, or play yourself.

## What is real
- NumPy forward/backward pass, Huber loss, Adam, replay memory and target network.
- 2,400 seeded training episodes on an 8×8 board.
- 100 held-out evaluation seeds: mean food **15.34**, median **15.5**, best **27**. Uniform-random baseline mean **0.06**.
- Python and browser environment/inference agree on all 100 evaluation scores.
- No action masking, heuristic rescue, remote model API, account, tracking, or GPU requirement.

## Limits
The network sees 11 engineered binary features, not pixels or the complete board. It cannot reliably reason about body traps. Reward includes a small distance-shaping term. Scores apply to this particular board, reward and move-limit configuration, not to every Snake implementation. Watching does not train or change the checkpoint.

## Reproduce
Python 3.10+ and NumPy. A CPU is sufficient.

```sh
python -m pip install -r requirements.txt
python snake.py --episodes 2400 --out model.json
python -m unittest -v
python -m http.server 8000
```

Open `http://localhost:8000` for development. The hosted browser app needs no installation.

`model.json` includes weights, full training history, seed list and every evaluation score. `snake.py` implements training; `engine.js` mirrors the environment and inference. Seeded food generation uses the same 32-bit LCG in both languages. Model initialization and training use seed 731, while training environment seeds start at 100. Evaluation seeds 20000–20099 are separate. Floating-point and NumPy-version differences may change a retrained policy.

## Method
The network is 11 → 64 (ReLU) → 3, 963 parameters. Relative actions: straight, right, left. Double DQN, replay memory 30,000, batch 64 every four environment steps, target sync every 400 steps, Adam 0.001, discount 0.95. Epsilon decreases linearly to 0.04 over the first 70% of episodes.

Reward: +10 food; −10 collision/timeout; −0.03 per step; ±0.08 distance shaping. Stop at 100 steps without food, 1,500 total steps, collision or a full board. Moving into the departing tail is legal. Food collected on exactly the total move limit still earns +10, then the episode ends. Greedy action selection uses the first maximum (straight, then right, then left) when Q-values tie.

Reference: [Human-level control through deep reinforcement learning, Mnih et al. (2015)](https://www.nature.com/articles/nature14236). This compact engineered-feature experiment is not a reproduction of the Atari paper.

## Privacy and security
Static HTML, CSS and JavaScript. The model is fetched from the same origin. No uploads, cookies, analytics, user accounts, storage or secrets. All displayed interactive values use text nodes. Run your own review before extending it to accept untrusted checkpoints or uploaded code.
