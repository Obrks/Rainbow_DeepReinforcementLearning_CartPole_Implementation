"""
Noisy DQN for CartPole-v1
Based on Fortunato et al. (2017) / Rainbow (Hessel et al. 2018)

What changes vs  DQN
  1. All nn.Linear layers are replaced with NoisyLinear layers.
     Each NoisyLinear layer has learnable mean (μ) and sigma (σ) parameters,
     and injects factorised Gaussian noise during the forward pass:

         y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)

     The network learns when to rely on noise and when to ignore it,
     producing state-conditional exploration that ε-greedy cannot.
    ε-greedy is removed entirely*

Usage 
    from noisy_dqn_cartpole import NoisyDQNConfig, NoisyDQNAgent, train, plot_results

    # 1. Train with default settings
    results = train()

    # 2. Train with custom config
    cfg     = NoisyDQNConfig(max_frames=100_000, sigma_init=0.4)
    results = train(cfg)

    # 3. Plot the score-vs-frame curve
    plot_results(results)

    # 4. Use the agent directly
    import gymnasium as gym
    env   = gym.make("CartPole-v1")
    agent = NoisyDQNAgent(env, NoisyDQNConfig())
    agent.save("noisy_dqn_cartpole.pt")

    # 5. Load a saved agent and evaluate
    agent2 = NoisyDQNAgent(env, NoisyDQNConfig())
    agent2.load("noisy_dqn_cartpole.pt")
    score = agent2.evaluate(env, n_episodes=10)
    print(f"Mean score: {score:.1f}")
"""

import math
import random
import time
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class NoisyDQNConfig:
    """
    All hyperparameters for the Noisy DQN agent and training loop
    """
    #Network
    learning_rate:      float = 1e-3
    hidden_size:        int   = 128
    sigma_init:         float = 0.5    # initial σ for noisy layers (Rainbow: 0.5)

    #Replay buffer
    replay_capacity:    int   = 50_000
    min_replay_size:    int   = 1_000
    batch_size:         int   = 64

    # RL 
    gamma:              float = 0.99
    target_update:      int   = 1_000
    train_freq:         int   = 1

    # Training loop
    max_frames:         int   = 200_000
    log_interval:       int   = 5_000
    solve_threshold:    float = 475.0

    render:             bool  = False
    device:             str   = "cpu"
    env_id:             str   = "CartPole-v1"


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """
    Circular experience replay buffer with uniform random sampling same as other models
    """

    def __init__(self, capacity: int):
        self.memory: deque = deque(maxlen=capacity)

    def push(self, state, action, reward: float, next_state, done: float) -> None:
    
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

class NoisyLinear(nn.Module):
    """
    Noisy linear layer

    layer = NoisyLinear(128, 64, sigma_init=0.5)
    y = layer(x)          # noisy forward pass
    layer.reset_noise()   # resample ε before each training step
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_init   = sigma_init

        # Learnable parameters
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # Noise resampled every step
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.empty(out_features))

        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self) -> None:
        """
        Initialise μ with uniform ± 1/√p and σ with σ_init / √p
        where p = in_features same as ref paper
        """
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """
        Resample factorised Gaussian noise called each step
        """
        eps_i = self._f(torch.randn(self.in_features,  device=self.weight_mu.device))
        eps_j = self._f(torch.randn(self.out_features, device=self.weight_mu.device))

        self.weight_epsilon.copy_(eps_j.outer(eps_i))
        self.bias_epsilon.copy_(eps_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Noisy forward pass
            y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)
        """
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"sigma_init={self.sigma_init}")

class NoisyDQNMlp(nn.Module):
    """
    Three-layer MLP Q-network with NoisyLinear layers
    Identical architecture to DQNMlp but all nn.Linear layers are replaced
    with NoisyLinear
    """

    def __init__(self, obs_size: int, n_actions: int,
                 hidden_size: int = 128, sigma_init: float = 0.5):
        super().__init__()
        self.fc1 = NoisyLinear(obs_size,    hidden_size, sigma_init)
        self.fc2 = NoisyLinear(hidden_size, hidden_size, sigma_init)
        self.fc3 = NoisyLinear(hidden_size, n_actions,   sigma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def reset_noise(self) -> None:
        """Resample noise in all three layers"""
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()



class NoisyDQNAgent:
    """
    DQN agent with Noisy Networks for exploration 
    """

    def __init__(self, env, cfg: NoisyDQNConfig):
        self.cfg       = cfg
        self.device    = torch.device(cfg.device)

        obs_size       = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.policy_net = NoisyDQNMlp(
            obs_size, self.n_actions, cfg.hidden_size, cfg.sigma_init
        ).to(self.device)
        self.target_net = NoisyDQNMlp(
            obs_size, self.n_actions, cfg.hidden_size, cfg.sigma_init
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.memory     = ReplayBuffer(cfg.replay_capacity)
        self.steps_done = 0

    

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action greedily no epsilon no random sampling
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(state_t).argmax(dim=1).item())


    def optimize_step(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one Noisy DQN gradient step
        """
        if len(self.memory) < self.cfg.min_replay_size:
            return None

        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        transitions = self.memory.sample(self.cfg.batch_size)
        batch       = Transition(*zip(*transitions))

        states      = torch.as_tensor(np.array(batch.state),      dtype=torch.float32, device=self.device)
        actions     = torch.tensor(batch.action,                   dtype=torch.long,    device=self.device).unsqueeze(1)
        rewards     = torch.tensor(batch.reward,                   dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones       = torch.tensor(batch.done,                     dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1).values
            targets = rewards + self.cfg.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self) -> None:
        """Copy policy network weights into the target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def mean_noise_magnitude(self) -> float:
        """
        Return the mean absolute value of the current noise parameters (σ)
        """
        sigmas = []
        for layer in [self.policy_net.fc1, self.policy_net.fc2, self.policy_net.fc3]:
            sigmas.append(layer.weight_sigma.abs().mean().item())
            sigmas.append(layer.bias_sigma.abs().mean().item())
        return float(np.mean(sigmas))

    def save(self, path: str) -> None:
        """
        Save the policy network weights and training state
        """
        torch.save({
            "policy_state":    self.policy_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "steps_done":      self.steps_done,
        }, path)
        print(f"Checkpoint saved → {path}")

    def load(self, path: str) -> None:
        """
        Load weights and training state from a checkpoint
        """
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.steps_done = ckpt["steps_done"]
        print(f"Checkpoint loaded ← {path}  (step {self.steps_done:,})")


    def evaluate(self, env, n_episodes: int = 10) -> float:
        """
        Run the greedy policy for n_episodes and return the mean score
        """
        scores = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            total    = 0.0
            done     = False
            while not done:
                state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action = int(self.policy_net(state_t).argmax(dim=1).item())
                state, reward, term, trunc, _ = env.step(action)
                total += reward
                done   = term or trunc
            scores.append(total)
        return float(np.mean(scores))


@dataclass
class TrainingResults:
    """
    data from the training run
    """
    frame_log:       list
    score_log:       list
    noise_log:       list          # extra: tracks σ decay over training
    episode_rewards: list
    solved_frame:    Optional[int]
    total_frames:    int
    total_episodes:  int
    elapsed_seconds: float
    agent:           NoisyDQNAgent



def train(cfg: Optional[NoisyDQNConfig] = None) -> TrainingResults:
    """
    Run the full Noisy DQN training loop and return a TrainingResults object
    """
    import gymnasium as gym

    if cfg is None:
        cfg = NoisyDQNConfig()

    print(" Noisy DQN ")
    print(f"  Replay buffer : {cfg.replay_capacity:,}")
    print(f"  Batch size    : {cfg.batch_size}")
    print(f"  Max frames    : {cfg.max_frames:,}")
    print("\n")

    render_mode = "human" if cfg.render else None
    env         = gym.make(cfg.env_id, render_mode=render_mode)
    agent       = NoisyDQNAgent(env, cfg)

    episode_rewards: list[float] = []
    frame_log:       list[int]   = []
    score_log:       list[float] = []
    noise_log:       list[float] = []

    ep_reward    = 0.0
    ep_count     = 0
    total_loss   = 0.0
    loss_steps   = 0
    solved       = False
    solved_frame = None

    state, _ = env.reset()
    t_start  = time.time()

    for frame in range(1, cfg.max_frames + 1):
        action                              = agent.select_action(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done                               = term or trunc

        agent.memory.push(state, action, float(reward), next_state, float(done))
        state        = next_state
        ep_reward   += reward
        agent.steps_done += 1

        if frame % cfg.train_freq == 0:
            loss = agent.optimize_step()
            if loss is not None:
                total_loss += loss
                loss_steps += 1

        if frame % cfg.target_update == 0:
            agent.sync_target()

        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            ep_count += 1
            state, _ = env.reset()

        if frame % cfg.log_interval == 0:
            recent  = episode_rewards[-100:] if episode_rewards else [0]
            mean_r  = float(np.mean(recent))
            mean_l  = total_loss / max(loss_steps, 1)
            noise   = agent.mean_noise_magnitude()
            fps     = frame / max(time.time() - t_start, 1e-6)
            elapsed = time.time() - t_start

            frame_log.append(frame)
            score_log.append(mean_r)
            noise_log.append(noise)

            bar_len  = 20
            progress = int(bar_len * frame / cfg.max_frames)
            bar      = "█" * progress + "░" * (bar_len - progress)

            print(
                f"[{bar}] {frame:>6,}/{cfg.max_frames:,}  "
                f"ep={ep_count:>4}  "
                f"mean_R={mean_r:>6.1f}  "
                f"σ={noise:.4f}  "
                f"loss={mean_l:.4f}  "
                f"fps={fps:.0f}  "
                f"{elapsed:.0f}s"
            )
            total_loss = 0.0
            loss_steps = 0

            if mean_r >= cfg.solve_threshold and len(recent) >= 100 and not solved:
                solved       = True
                solved_frame = frame
                print(f"\n  SOLVED at frame {frame:,} "
                      f"(mean reward {mean_r:.1f} over last 100 episodes)\n")

    env.close()
    elapsed = max(time.time() - t_start, 1e-6)
    print(f"\nDone. {ep_count} episodes in {elapsed:.0f}s ({frame / elapsed:.0f} fps avg)")

    return TrainingResults(
        frame_log       = frame_log,
        score_log       = score_log,
        noise_log       = noise_log,
        episode_rewards = episode_rewards,
        solved_frame    = solved_frame,
        total_frames    = frame,
        total_episodes  = ep_count,
        elapsed_seconds = elapsed,
        agent           = agent,
    )


def plot_results(results: TrainingResults) -> None:
    """
    Plot score-vs-frame and σ decay on a two-panel figure
    """
    import matplotlib.pyplot as plt

    cfg      = results.agent.cfg
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(results.frame_log, results.score_log,
               s=12, alpha=0.35, color="mediumseagreen", zorder=2,
               label="Mean score (100 ep window)")

    if len(results.score_log) >= 5:
        k        = min(5, len(results.score_log))
        smoothed = np.convolve(results.score_log, np.ones(k) / k, mode="valid")
        ax.plot(results.frame_log[k - 1:], smoothed,
                color="mediumseagreen", linewidth=2.5, zorder=3,
                label=f"Smoothed ({k}-point moving avg)")

    ax.axhline(cfg.solve_threshold, color="gold", linestyle="--",
               linewidth=1.5, zorder=1, label=f"Solve threshold ({cfg.solve_threshold})")

    if results.solved_frame is not None:
        ax.axvline(results.solved_frame, color="limegreen", linestyle=":",
                   linewidth=1.5, zorder=1,
                   label=f"Solved at frame {results.solved_frame:,}")

    ax.set_xlabel("Environment frames", fontsize=12)
    ax.set_ylabel("Mean episode score (last 100 ep)", fontsize=12)
    ax.set_title(f"Noisy DQN on {cfg.env_id} — Score vs Frames", fontsize=13)
    ax.set_xlim(0, cfg.max_frames)
    ax.set_ylim(0, 520)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(results.frame_log, results.noise_log,
             color="mediumseagreen", linewidth=2.0, label="Mean |σ| (all layers)")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)
    ax2.set_xlabel("Environment frames", fontsize=12)
    ax2.set_ylabel("Mean noise magnitude |σ|", fontsize=12)
    ax2.set_title("Noise Self-Annealing", fontsize=13)
    ax2.set_xlim(0, cfg.max_frames)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle("Noisy DQN — CartPole-v1", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()


