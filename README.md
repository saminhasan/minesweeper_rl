# Minesweeper RL — AI Decision-Making Under Uncertainty

A hybrid Minesweeper AI that chains an exact probabilistic solver, a CNN probability predictor, and a PPO reinforcement learning agent into a single end-to-end pipeline.



---

## Project Objective

Minesweeper is a partially observable puzzle in which the player must select actions from incomplete information. Exact probabilistic solvers can play strongly, but their inference cost grows exponentially with frontier complexity. This project investigates whether a CNN can learn a fixed-cost approximation of the solver's posterior mine-probability maps, and whether a reinforcement learning agent trained on those learned maps can approach solver-level gameplay performance.

The three-stage pipeline is:

```
Exact Bayesian Solver  →  CNN Probability Predictor  →  PPO RL Agent
   (dense targets)           (fast approximation)         (action selection)
```

---

## Supported Difficulty Levels

| Difficulty   | Rows | Columns | Mines |
|--------------|------|---------|-------|
| Easy         | 10   | 10      | 10    |
| Intermediate | 16   | 16      | 40    |
| Hard         | 16   | 30      | 99    |

---

## Repository Structure

```
minesweeper_rl/
├── game_engine.py            # Minesweeper game engine (reveal, flood-fill, frontier)
├── solver.py                 # Exact probabilistic Minesweeper solver
├── solver_util.py            # Solver utilities (supercells, rule reduction)
├── predictor.py              # Unified probability interface (Bayesian or CNN)
├── rl.py                     # RL policy network architecture (actor-critic)
│
├── cnn_dataset_gen.py        # Generate solver-labelled training data
├── cnn_train.py              # Train the CNN probability predictor
├── cnn_compare_prediction.py # Evaluate & compare Bayesian vs CNN predictions
│
├── rl_train.py               # Train the RL agent with PPO
├── rl_play.py                # Evaluate the RL agent over N games
├── rl_compare_play.py        # Head-to-head: RL vs Bayesian gameplay
├── bayes_play.py             # Evaluate the Bayesian solver over N games
│
├── models/
│   ├── CNN/{easy,intermediate,hard}/model.keras
│   └── RL/{easy,intermediate,hard}/policy.keras
│
├── data/                     # CNN training datasets (generated locally)
│   └── {easy,intermediate,hard}/
│
└── logs/                     # Evaluation outputs, plots, JSON aggregates
    └── {easy,intermediate,hard}/
```

---

## Installation

### Requirements

- Python 3.10+
- TensorFlow 2.x (GPU recommended)
- NumPy, tqdm, matplotlib

```bash
pip install tensorflow numpy tqdm matplotlib
```

### Clone

```bash
git clone https://github.com/saminhasan/minesweeper_rl.git
cd minesweeper_rl
```

---

## Usage

The pipeline has four ordered stages. Each stage can be run independently if the prior stage's outputs already exist.

### Stage 1 — Generate CNN Training Data

Plays Minesweeper games using the exact Bayesian solver and records per-step board snapshots (visible state + posterior mine-probability map + covered mask) as binary training data.

```bash
python cnn_dataset_gen.py
```

Configure at the top of the file:

| Variable       | Default        | Description                    |
|----------------|----------------|--------------------------------|
| `ACTIVE_LEVEL` | `"easy"`       | Difficulty to generate data for |
| `N_GAMES`      | `262144`       | Number of games to simulate    |
| `SAFE_START`   | `True`         | Reveal one safe cell first     |
| `OUT_DIR`      | `data/<level>` | Where to save the binary dataset |

Output: `data/<level>/<timestamp>/data.bin` + `metadata.json`

---

### Stage 2 — Train the CNN

Trains a fully-convolutional residual-dilated network to predict per-cell mine and safe probabilities from the four-channel board representation.

```bash
python cnn_train.py
```

Configure `ACTIVE_LEVEL` and `LEVEL_CONFIGS` at the bottom of the file.

Key hyperparameters (file top):

| Variable                 | Default | Description                                  |
|--------------------------|---------|----------------------------------------------|
| `LEARNING_RATE`          | `1e-3`  | Adam learning rate                           |
| `COVERED_WEIGHT`         | `2.0`   | Loss weight for covered cells                |
| `UNDERESTIMATION_PENALTY`| `2.0`   | Extra penalty for under-predicting mine prob |
| `EARLY_STOPPING_PATIENCE`| `5`     | Epochs without improvement before stopping  |
| `BASE_CHANNELS`          | `64`    | Base channel width (level-specific)          |

The model is saved to `models/CNN/<level>/model.keras` only when validation loss improves. Training curves are saved to `models/CNN/<level>/training_curves/`.

---

### Stage 3 — Train the RL Agent

Trains a PPO actor-critic policy using the six-channel state:

```
[board_scaled | covered_mask | revealed_clue | frontier_mask | P̂(mine) | P̂(safe)]
```

```bash
python rl_train.py
```

Configure `ACTIVE_LEVEL` and `LEVEL_CONFIGS` at the bottom of the file.

Key hyperparameters:

| Variable              | Default | Description                            |
|-----------------------|---------|----------------------------------------|
| `LEARNING_RATE`       | `1e-4`  | Adam learning rate                     |
| `GAMMA`               | `0.99`  | Discount factor                        |
| `GAE_LAMBDA`          | `0.95`  | GAE advantage estimation lambda        |
| `PPO_CLIP_EPS`        | `0.2`   | PPO clipping epsilon                   |
| `ENTROPY_BETA`        | `1e-2`  | Entropy bonus coefficient              |
| `GREEDY_ACTION_PROB`  | `0.5`   | Fraction of steps using greedy action  |

The policy is saved to `models/RL/<level>/policy.keras` when win rate improves.

---

### Stage 4 — Evaluate

#### Bayesian solver baseline

```bash
python bayes_play.py
```

#### RL agent

```bash
python rl_play.py
```

#### Head-to-head comparison (RL vs Bayesian)

Plays both agents on identical random seeds and produces a comparison plot and CSV.

```bash
python rl_compare_play.py
```

#### CNN prediction quality (Bayesian vs CNN)

Evaluates Brier score, FPR, FNR, and ROC-AUC on held-out games.

```bash
python cnn_compare_prediction.py
```

All scripts write results to `logs/<level>/` with timestamped subdirectories containing a CSV, `aggregate.json`, and plots.

---

## Trained Models

Pre-trained models are included in the repository under `models/`:

```
models/CNN/easy/model.keras
models/CNN/intermediate/model.keras
models/CNN/hard/model.keras

models/RL/easy/policy.keras
models/RL/intermediate/policy.keras
models/RL/hard/policy.keras
```

To use the trained models directly, run any Stage 4 evaluation script — they load the model automatically from the `models/` path.

---

## Key Findings

### CNN Prediction Quality

The CNN tracked the exact Bayesian solver closely across all difficulty levels. Prediction quality degrades as board difficulty increases because harder boards contain larger ambiguous frontiers.

| Level        | Brier Score | FPR  | FNR  | ROC-AUC |
|--------------|-------------|------|------|---------|
| Easy         | 0.04        | 0.01 | 0.19 | 0.94    |
| Intermediate | 0.09        | 0.01 | 0.31 | 0.89    |
| Hard         | 0.14        | 0.01 | 0.43 | 0.81    |

The CNN achieved **lower FPR** than the Bayesian baseline thresholded at 0.5 (more conservative about flagging safe cells as mines), consistent with the asymmetric training loss that penalizes mine underestimation. FNR — the safety-critical metric, since a false negative means treating a mine as safe — remained higher than the Bayesian teacher, but the gap was modest on easy and intermediate boards.

### RL Gameplay Results

The RL agent was compared against the exact Bayesian baseline on matched seeds.

| Level        | Agent    | Win Rate | Avg. Steps |
|--------------|----------|----------|------------|
| Easy         | Bayesian | 88.0%    | 14.24      |
|              | RL       | **89.0%**| 15.38      |
| Intermediate | Bayesian | **66.5%**| 59.66      |
|              | RL       | 66.0%    | 65.00      |
| Hard         | Bayesian | **36.0%**| 134.15     |
|              | RL       | 21.0%    | 108.00     |

The RL policy matched or exceeded the Bayesian solver on easy boards and was competitive on intermediate. On hard boards the exact solver remained substantially stronger — shorter RL episode lengths on hard reflect earlier mine hits, not faster solutions.

### Interpretation

- A learned CNN can faithfully approximate exact solver probability maps at **constant inference cost** regardless of frontier complexity.
- A PPO policy operating on uncertainty-aware state achieves **near-Bayesian gameplay** on easy and intermediate difficulty.
- Hard boards — where the frontier is large and clues under-constrain the hidden state — still benefit most from exact probabilistic reasoning.

---

## Architecture Details

### CNN

- 4-channel input: `[board_scaled, covered_mask, revealed_clue, frontier_mask]`
- Stack of residual dilated convolutional blocks (dilation rates `1, 2, 4, 8`)
- Final 2-channel softmax head: `[P̂(mine), P̂(safe)]`
- Asymmetric weighted loss: covered cells × 2, mine underestimation × 2

### RL Policy

- 6-channel input: base 4 channels + `[P̂(mine), P̂(safe)]` from CNN
- Actor-critic with shared convolutional body and separate policy/value heads
- Action masking: only currently covered cells are selectable
- Reward: `-1` on mine hit; `Δuncovered/U + 0.5·Δfrontier/U + 1` on win

---

## Reproducing Results

```bash
# 1. Generate data (easy level, ~260k games)
python cnn_dataset_gen.py

# 2. Train CNN
python cnn_train.py

# 3. Evaluate CNN predictions
python cnn_compare_prediction.py

# 4. Train RL
python rl_train.py

# 5. Compare RL vs Bayesian
python rl_compare_play.py
```

Change `ACTIVE_LEVEL` at the top (or bottom) of each script to `"intermediate"` or `"hard"` and repeat.


