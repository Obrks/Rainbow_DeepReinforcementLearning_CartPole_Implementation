"""
DQN with Prioritized Experience Replay (PER) for CartPole
Based on Schaul et al. (2015) / Rainbow (Hessel et al. 2018)

changes vs DQN

  Simple DQN samples transitions uniformly at random from the replay buffer
  PER samples transitions proportional to their TD error transitions the
  agent has the most to learn from are replayed more frequently
  
  ReplayBuffer → PrioritizedReplayBuffer
     New transitions are inserted with maximum priority so they are sampled
     at least once before their TD error is known

  optimize_step() returns TD errors and feeds them back to update priorities
     Sampling returns indices and importance-sampling weights 

  hyperparameters:
       alpha (α): controls how much prioritization is used.
                  α=0 → uniform (simple DQN), α=1 → fully prioritized
       beta  (β): importance-sampling exponent 
                  Corrects the bias from non-uniform sampling
"""

import random
import time
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class PERConfig:
    """
    parameters for the PER DQN

    PER  parameters:
        alpha:      prioritization exponent 0 = uniform, 1 = fully prioritized
        beta_start: initial importance-sampling exponent 
        per_eps:    small constant added to every priority to prevent zero
                    probability: p = |δ|^α + ε.
    """
    #Network  
    learning_rate:      float = 1e-3
    hidden_size:        int   = 128

    #Replay buffer
    replay_capacity:    int   = 50_000
    min_replay_size:    int   = 1_000
    batch_size:         int   = 64

    #Prioritized replay 
    alpha:              float = 0.6      
    beta_start:         float = 0.4      
    per_eps:            float = 1e-5     

    #RL 
    gamma:              float = 0.99
    target_update:      int   = 1_000
    train_freq:         int   = 1

    # Exploration
    eps_start:          float = 1.0
    eps_end:            float = 0.01
    eps_decay_frames:   int   = 10_000

    #Training loop
    max_frames:         int   = 200_000
    log_interval:       int   = 5_000
    solve_threshold:    float = 475.0

    # Misc 
    render:             bool  = False
    device:             str   = "cpu"
    env_id:             str   = "CartPole-v1"


class SumTree:
    """
    Binary sum-tree for priority sampling and updates

    Each leaf stores the priority p_i of one transition, every internal node stores
    the sum of its two children and the root stores the total sum of all priorities

    Sampling a transition proportional to its priority p_i / Σ p_j is done by
    drawing a uniform value u ~ U[0, Σ p_j] and traversing the tree 

    
        tree = SumTree(capacity=1000)
        tree.add(priority=1.0, data=transition)
        indices, priorities, batch = tree.sample(32)
        tree.update(indices, new_priorities)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data     = np.empty(capacity, dtype=object)
        self.write    = 0          # circular write pointer
        self.n_entries = 0

    def _propagate(self, idx: int, delta: float) -> None:
        """Propagate a priority change up the tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find the leaf index whose cumulative priority spans value"""
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        return self._retrieve(right, value - self.tree[left])

    @property
    def total(self) -> float:
        """Total sum of all priorities, root node"""
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        """Insert a new transition with the given priority """
        leaf_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(leaf_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        """Update the priority of an existing leaf """
        delta               = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def sample(self, batch_size: int):
        """
        Sample batch_size transitions proportional to their priorities
        """
        indices    = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float64)
        data       = []

        segment = self.total / batch_size
        for i in range(batch_size):
            lo  = segment * i
            hi  = segment * (i + 1)
            val = random.uniform(lo, hi)
            idx = self._retrieve(0, val)
            indices[i]    = idx
            priorities[i] = self.tree[idx]
            data.append(self.data[idx - self.capacity + 1])

        return indices, priorities, data

    def __len__(self) -> int:
        return self.n_entries


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class PrioritizedReplayBuffer:
    """
    Experience replay buffer with prioritized sampling (Schaul et al. 2015)

    Transitions are sampled with probability proportional to:

        p_i = (|δ_i| + ε)^α

    where δ_i is the TD error of transition i, ε is a small floor constant,
    and α ∈ [0, 1] controls the degree of prioritization

    To correct the bias introduced by non-uniform sampling, each sampled
    transition is weighted by the importance-sampling (IS) weight:

        w_i = (1 / N · 1 / P(i))^β

    normalized by max_i w_i so that the maximum weight is always 1.
    β is annealed from beta_start → 1 over the course of training

    """

    def __init__(self, capacity: int, alpha: float,
                 beta_start: float, per_eps: float):
        self.tree      = SumTree(capacity)
        self.alpha     = alpha
        self.beta      = beta_start
        self.per_eps   = per_eps
        self.max_prio  = 1.0      

    def push(self, state, action, reward: float, next_state, done: float) -> None:
        """Store one transition with maximum current priority """
        transition = Transition(state, action, reward, next_state, done)
        self.tree.add(self.max_prio ** self.alpha, transition)

    def sample(self, batch_size: int, beta: float):
        """
        Sample a prioritized mini-batch
        """
        indices, priorities, transitions = self.tree.sample(batch_size)

        # Importance-sampling weights
        N        = len(self.tree)
        probs    = priorities / self.tree.total
        weights  = (N * probs) ** (-beta)
        weights /= weights.max()           # normalise so max weight = 1

        return indices, weights.astype(np.float32), transitions

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities after a gradient step using fresh TD errors

        """
        for idx, err in zip(indices, td_errors):
            prio           = (abs(err) + self.per_eps) ** self.alpha
            self.max_prio  = max(self.max_prio, prio)
            self.tree.update(int(idx), prio)

    def __len__(self) -> int:
        return len(self.tree)


class PERMlp(nn.Module):
    """
    Three-layer MLP Q-network same as others

    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PERAgent:
    """
    DQN agent with Prioritized Experience Replay 
    """

    def __init__(self, env, cfg: PERConfig):
        self.cfg       = cfg
        self.device    = torch.device(cfg.device)

        obs_size       = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.policy_net = PERMlp(obs_size, self.n_actions, cfg.hidden_size).to(self.device)
        self.target_net = PERMlp(obs_size, self.n_actions, cfg.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.memory     = PrioritizedReplayBuffer(
                              cfg.replay_capacity, cfg.alpha,
                              cfg.beta_start, cfg.per_eps)
        self.steps_done = 0

    

    def beta(self) -> float:
        progress = min(self.steps_done / self.cfg.max_frames, 1.0)
        return self.cfg.beta_start + (1.0 - self.cfg.beta_start) * progress


    def epsilon(self) -> float:
        """Current ε for ε-greedy"""
        progress = min(self.steps_done / self.cfg.eps_decay_frames, 1.0)
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * (1.0 - progress)

    def select_action(self, state: np.ndarray, env) -> int:
        """
        ε-greedy action selection same as simpla DQN
        """
        if random.random() < self.epsilon():
            return env.action_space.sample()
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(state_t).argmax(dim=1).item())


    def optimize_step(self) -> Optional[float]:
        """
        Sample a prioritized mini-batch and perform one gradient step
        """
        if len(self.memory) < self.cfg.min_replay_size:
            return None

        beta                          = self.beta()
        indices, weights, transitions = self.memory.sample(self.cfg.batch_size, beta)
        batch                         = Transition(*zip(*transitions))

        states      = torch.as_tensor(np.array(batch.state),      dtype=torch.float32, device=self.device)
        actions     = torch.tensor(batch.action,                   dtype=torch.long,    device=self.device).unsqueeze(1)
        rewards     = torch.tensor(batch.reward,                   dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones       = torch.tensor(batch.done,                     dtype=torch.float32, device=self.device)
        weights_t   = torch.as_tensor(weights,                     dtype=torch.float32, device=self.device)

        # Q(s, a) from policy network
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # TD target from target network
        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1).values
            targets = rewards + self.cfg.gamma * next_q * (1.0 - dones)

        # PER IS-weighted element-wise Huber loss
        td_errors    = (targets - q_values).detach().cpu().numpy()
        element_loss = F.smooth_l1_loss(q_values, targets, reduction="none")
        loss         = (weights_t * element_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        #feed fresh TD errors back to the buffer 
        self.memory.update_priorities(indices, np.abs(td_errors))
        

        return loss.item()

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())


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
    data from the training
    """
    frame_log:       list
    score_log:       list
    beta_log:        list
    episode_rewards: list
    solved_frame:    Optional[int]
    total_frames:    int
    total_episodes:  int
    elapsed_seconds: float
    agent:           PERAgent


def train(cfg: Optional[PERConfig] = None) -> TrainingResults:
    """
    Run the full PER DQN training loop and return a TrainingResults object
    """
    import gymnasium as gym

    if cfg is None:
        cfg = PERConfig()

    print("PERDQN - CartPole")
    print(f"  Replay buffer : {cfg.replay_capacity:,}")
    print(f"  Batch size    : {cfg.batch_size}")
    print(f"  Max frames    : {cfg.max_frames:,}")
    print("\n")

    render_mode = "human" if cfg.render else None
    env         = gym.make(cfg.env_id, render_mode=render_mode)
    agent       = PERAgent(env, cfg)

    episode_rewards: list[float] = []
    frame_log:       list[int]   = []
    score_log:       list[float] = []
    beta_log:        list[float] = []

    ep_reward    = 0.0
    ep_count     = 0
    total_loss   = 0.0
    loss_steps   = 0
    solved       = False
    solved_frame = None

    state, _ = env.reset()
    t_start  = time.time()

    for frame in range(1, cfg.max_frames + 1):
        action                              = agent.select_action(state, env)
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
            beta    = agent.beta()
            fps     = frame / max(time.time() - t_start, 1e-6)
            elapsed = time.time() - t_start

            frame_log.append(frame)
            score_log.append(mean_r)
            beta_log.append(beta)

            bar_len  = 20
            progress = int(bar_len * frame / cfg.max_frames)
            bar      = "█" * progress + "░" * (bar_len - progress)

            print(
                f"[{bar}] {frame:>6,}/{cfg.max_frames:,}  "
                f"ep={ep_count:>4}  "
                f"mean_R={mean_r:>6.1f}  "
                f"eps={agent.epsilon():.3f}  "
                f"β={beta:.3f}  "
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
        beta_log        = beta_log,
        episode_rewards = episode_rewards,
        solved_frame    = solved_frame,
        total_frames    = frame,
        total_episodes  = ep_count,
        elapsed_seconds = elapsed,
        agent           = agent,
    )

def plot_results(results: TrainingResults) -> None:
    """
    Plot score-vs-frame and β annealing on a two-panel figure

    The second panel shows β increasing from beta_start → 1.0 over training,
    reflecting the growing correction for the prioritization bias
    """
    import matplotlib.pyplot as plt

    cfg       = results.agent.cfg
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(results.frame_log, results.score_log,
               s=12, alpha=0.35, color="mediumpurple", zorder=2,
               label="Mean score (100 ep window)")

    if len(results.score_log) >= 5:
        k        = min(5, len(results.score_log))
        smoothed = np.convolve(results.score_log, np.ones(k) / k, mode="valid")
        ax.plot(results.frame_log[k - 1:], smoothed,
                color="mediumpurple", linewidth=2.5, zorder=3,
                label=f"Smoothed ({k}-point moving avg)")

    ax.axhline(cfg.solve_threshold, color="gold", linestyle="--",
               linewidth=1.5, zorder=1, label=f"Solve threshold ({cfg.solve_threshold})")

    if results.solved_frame is not None:
        ax.axvline(results.solved_frame, color="limegreen", linestyle=":",
                   linewidth=1.5, zorder=1,
                   label=f"Solved at frame {results.solved_frame:,}")

    ax.set_xlabel("Environment frames", fontsize=12)
    ax.set_ylabel("Mean episode score (last 100 ep)", fontsize=12)
    ax.set_title(f"PER DQN on {cfg.env_id} — Score vs Frames", fontsize=13)
    ax.set_xlim(0, cfg.max_frames)
    ax.set_ylim(0, 520)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(results.frame_log, results.beta_log,
             color="mediumpurple", linewidth=2.0, label="β (IS exponent)")
    ax2.axhline(1.0, color="gold", linestyle="--", linewidth=1.0,
                alpha=0.7, label="β = 1.0 (full correction)")
    ax2.set_xlabel("Environment frames", fontsize=12)
    ax2.set_ylabel("β", fontsize=12)
    ax2.set_title("Importance-Sampling β Annealing", fontsize=13)
    ax2.set_xlim(0, cfg.max_frames)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle("PER DQN — CartPole-v1", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

