# Rainbow Deep Q-Network — CartPole Implementation

> An academic implementation of the Rainbow reinforcement learning algorithm (Hessel et al. 2018), built incrementally from vanilla DQN to the full combined agent. Each component is implemented as a standalone, importable Python module.
<p align="center">
  <img src="rainbow_cartpole.gif" width="400" alt="Rainbow agent solving CartPole"/>
  ![Rainbow CartPole](rainbow_cartpole.gif)
  <br>
  <em>Trained Rainbow agent balancing the pole</em>
</p>
This project was developed as part of an Reinforcement Learning project at Université Paris Dauphine — Master MASEF. The goal is to reproduce the results of the Rainbow paper by implementing each of the six DQN improvements individually, studying their individual contributions, and combining them into a single agent. The implementation is on the CartPole game for the sake of simplicity and not 57 Atari games.  

## What is Rainbow?

Rainbow combines six independent improvements to the original Deep Q-Network (Mnih et al. 2015) into a single agent. Each component targets a specific weakness of the original DQN:

| Component | Paper | What it fixes |
|---|---|---|
| **Double DQN** | van Hasselt et al. (2016) | Overestimation of Q-values |
| **Prioritized Replay** | Schaul et al. (2015) | Wasteful uniform experience sampling |
| **Noisy Nets** | Fortunato et al. (2017) | Undirected ε-greedy exploration |
| **Multi-Step Returns** | Sutton (1988) | Slow reward propagation through time |
| **Distributional RL** | Bellemare et al. (2017) | Discarding return variance |
| **Dueling Networks** | Wang et al. (2016) | Poor estimates for irrelevant actions |

The key finding of the paper is that no single component accounts for the full performance gain, the improvements are complementary and interact positively when combined.

## Project Structure

```
├── DQNModel.py              # Vanilla DQN baseline
├── DoubleQModel.py       # Double DQN
├── PreplayModel.py          # Prioritized Experience Replay
├── NoisyModel.py        # Noisy Networks
├── MultiStepModel.py    # Multi-Step Returns
├── DistributionModel.py              # Distributional RL 
├── DuelingModel.py      # Dueling Networks
├── RainbowModel.py          # Rainbow (all except Dueling)
│
├── Trained Models/              # Saved .pt checkpoints
│
└──References/
```

---

## Environment

All agents are trained and evaluated on CartPole (Gymnasium) a classic control benchmark where a pole must be balanced on a moving cart. The task is considered solved when the agent achieves a mean episode reward ≥ 475 over 100 consecutive episodes (maximum possible reward per episode: 500).

The environment was chosen for its speed on CPU with no GPU required...

```pip install gymnasium torch numpy matplotlib imageio```

## Quick Start

Every module exposes the same  Training, evaluating, plotting. The same pattern works for every module, just swap the import.

```python
# Train
from DQNModel import DQNConfig, train, plot_results

results = train(DQNConfig(max_frames=200_000))
plot_results(results)
results.agent.save("Trained Models/dqn.pt")

# Load and evaluate
import gymnasium as gym
from dqn_cartpole import DQNAgent, DQNConfig

env   = gym.make("CartPole-v1")
agent = DQNAgent(env, DQNConfig())
agent.load("Trained Models/dqn.pt")
print(agent.evaluate(env, n_episodes=20))

# Watch the agent play
env = gym.make("CartPole-v1", render_mode="human")
agent.evaluate(env, n_episodes=3)
env.close()
```


## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature 518.
- van Hasselt, Guez & Silver (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI-16.
- Schaul et al. (2015). *Prioritized Experience Replay*. ICLR 2016.
- Fortunato et al. (2017). *Noisy Networks for Exploration*. ICLR 2018.
- Sutton (1988). *Learning to Predict by the Methods of Temporal Differences*. Machine Learning 3(1).
- Bellemare, Dabney & Munos (2017). *A Distributional Perspective on Reinforcement Learning*. ICML 2017.
- Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. ICML 2016.
- Hessel et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning*. AAAI-18.



*Université Paris Dauphine — Master MASEF — Reinforcement Learning*
