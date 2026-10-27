
# Treasure Hunter: Learning Optimal Navigation Using Deep Q-Networks

## Overview

Treasure Hunter is a reinforcement learning mini-project in which an autonomous agent learns to collect a treasure, avoid traps and obstacles, and reach an exit in a grid-based environment.

The agent uses a Deep Q-Network (DQN) implemented in PyTorch. It learns from rewards and previous experiences, and its performance is evaluated using training curves, success rate, average reward, and steps required to complete the task.


<img width="340" height="327" alt="image" src="https://github.com/user-attachments/assets/c171fd4b-aeb7-43ea-98e5-df53c86cf5d0" />

## Features

- Custom 6 × 6 grid environment
- Four movement actions: up, down, left, and right
- Reward-based learning
- DQN with experience replay and a target network
- Keyboard-controlled Pygame game
- Autonomous AI gameplay
- Training and evaluation graphs
- Comparison of two epsilon-decay settings

## Technologies

- Python 3.11
- PyTorch
- NumPy
- Pygame
- Matplotlib

## Project Structure

- `environment.py` — game environment, state, actions, rewards, and termination
- `dqn_agent.py` — DQN neural network, replay buffer, and learning logic
- `train.py` — model training and creation of training graphs
- `evaluate.py` — evaluation of a trained model
- `game.py` — keyboard-controlled game
- `ai_game.py` — autonomous gameplay using the trained model
- `test_environment.py` — environment tests
- `test_dqn.py` — DQN component tests
- `compare_experiments.py` — comparison of epsilon-decay experiments
- `models/` — saved neural-network weights
- `results/` — training data, graphs, and evaluation records

## Reinforcement Learning Design

**State:** 37 numerical values representing the grid and treasure status.

**Actions:** Up, down, left, and right.

**Rewards:**
- Normal movement: -0.1
- Treasure collected: +10
- Exit reached after collecting treasure: +20
- Trap entered: -10
- Wall or boundary collision: -2

**DQN architecture:** 37 input values, two hidden layers of 128 neurons, and four output Q-values.

## Installation

Use Python 3.11 and run the following commands in the project directory.

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## How to Run

Train the agent:

```powershell
python train.py
```

Evaluate the trained model:

```powershell
python evaluate.py
```

Play manually:

```powershell
python game.py
```

Watch the trained AI play:

```powershell
python ai_game.py
```

Compare experiments:

```powershell
python compare_experiments.py
```

Run the environment tests:

```powershell
python test_environment.py
```

Test the DQN:

```powershell
python test_dqn.py
```

## Results

The original trained model achieved the following results on the fixed map:
<img width="1500" height="750" alt="epsilon_reward_comparison" src="https://github.com/user-attachments/assets/2629c45a-010b-4fcc-96b4-e1318a3d0604" />
## epsilon-decay reward comparison
<img width="1500" height="750" alt="epsilon_success_comparison" src="https://github.com/user-attachments/assets/d5aac6a6-d1f2-4cc4-80a0-9aa9f9ddb1f6" />
## epsilon-decay success comparison



| Metric | Result |
|---|---:|
| Training episodes | 1,000 |
| Evaluation episodes | 30 |
| Successful evaluation episodes | 30 |
| Evaluation success rate | 100% |
| Average evaluation reward | 29.20 |
| Average evaluation steps | 10 |

The project also investigates the effect of epsilon-decay settings on training reward and success rate. See the graphs saved in the `results/` directory.

These results apply to the fixed environment and do not establish generalisation to unseen maps.

## Limitations and Future Work

The environment uses a fixed map and starting position. Future improvements include randomised maps, additional treasures, moving obstacles, multiple training seeds, and comparison with a separate baseline algorithm.

## Author

Student Name: VAISHNAVI THAKUR

Course: Reinforcement Learning

Institution: Manipal University Jaipur
