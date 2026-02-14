# CarAI -- Self-Driving Car with PPO + LSTM

A neural network learns to drive a car around increasingly difficult circuits using **Proximal Policy Optimization (PPO)** with an **LSTM Actor-Critic** architecture and **SquashedNormal** continuous action distribution.

## Architecture

```
Inputs (15) : 12 radars (360 deg) + speed + angle_to_checkpoint + curvature
Network     : FC(128) -> FC(128) -> LSTM(128) -> Actor(2) + Critic(1)
Outputs (2) : steering [-1, +1], throttle [-1, +1]
Distribution: SquashedNormal (tanh-squashed Gaussian with Jacobian correction)
Training    : PPO with GAE, sequential LSTM mini-batches (chunks of 32)
Obs Norm    : Welford running normaliser
Parameters  : ~167k
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Usage

### Train (new run)
```bash
python main.py
```

### Resume training from checkpoint
```bash
python main.py --resume
python main.py --checkpoint models/checkpoint.pt
```

### Test a trained model
```bash
python main.py --test
python main.py --test models/checkpoint.pt
```

## Controls

| Key       | Action                     |
|-----------|----------------------------|
| `SPACE`   | Pause / Resume             |
| `R`       | Toggle radars              |
| `S`       | Save checkpoint (train)    |
| `N`       | Next test track (test)     |
| `Up/Down` | Simulation speed           |
| `ESC`     | Quit (auto-saves)          |

## Training Levels

The car progresses through 5 difficulty levels after completing 2 laps each:
1. Oval (easy)
2. Simple curves
3. Chicanes
4. Complex circuit
5. Expert track

## Checkpoints

Checkpoints are saved as `.pt` files via `torch.save` and include:
- Model weights (policy + critic)
- Optimizer state
- Episode / global step counters
- Observation normaliser statistics
- Training stats (best fitness, recent rewards)

## Project Structure

| File       | Description                            |
|------------|----------------------------------------|
| `main.py`  | Entry point, train/test loops, CLI     |
| `brain.py` | ActorCriticLSTM, PPO, SquashedNormal   |
| `car.py`   | Car physics, sensors, reward shaping   |
| `track.py` | 5 training levels + 3 test tracks      |
| `gui.py`   | Dual-window GUI (track + NN viz)       |
