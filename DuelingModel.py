"""
Dueling DQN for CartPole
Based on Wang et al. (2016) / Rainbow (Hessel et al. 2018)

changes 
  Vanilla DQN uses a single stream to estimate Q(s, a) directly
  Dueling DQN splits the network into two separate streams after the
  shared feature layers:

      Value stream:     V(s)         — how good is this state, regardless of action
      Advantage stream: A(s, a)      — how much better is action a vs the average

  They are recombined via the identifiable aggregation formula:

      Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

  Subtracting the mean advantage forces V(s) to represent the true state value

  This architecture helps in states where the choice of action does not
  matter much the value stream can learn V(s) accurately without
  needing to evaluate every action, leading to better Q estimates
  especially for rarely-visited actions

  Only the network architecture changes. The training loop, replay buffer,
  target network, epsilon-greedy, and loss function are all identical to
  vanilla DQN. This makes dueling a near-free improvement

  One new hyperparameter:
      advantage_hidden_size: units in each advantage/value head (default 128).
"""

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
class DuelingConfig:
    """
    parameters for the Dueling DQN agent and training loop
    """
    #Network
    learning_rate:          float = 1e-3
    hidden_size:            int   = 128   # shared trunk hidden size
    advantage_hidden_size:  int   = 128   # per-head hidden size

    # Replay buffer
    replay_capacity:        int   = 50_000
    min_replay_size:        int   = 1_000
    batch_size:             int   = 64

    #RL 
    gamma:                  float = 0.99
    target_update:          int   = 1_000
    train_freq:             int   = 1

    # Exploration 
    eps_start:              float = 1.0
    eps_end:                float = 0.01
    eps_decay_frames:       int   = 10_000

    # Training loop 
    max_frames:             int   = 200_000
    log_interval:           int   = 5_000
    solve_threshold:        float = 475.0

    # Misc 
    render:                 bool  = False
    device:                 str   = "cpu"
    env_id:                 str   = "CartPole-v1"


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """Circular replay buffer with uniform random sampling"""

    def __init__(self, capacity: int):
        self.memory: deque = deque(maxlen=capacity)

    def push(self, state, action, reward: float, next_state, done: float) -> None:
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DuelingMlp(nn.Module):
    """
    Dueling MLP Q-network 

    Architecture:
        Shared trunk  : Linear(obs) → ReLU → Linear → ReLU
                        Extracts a feature representation common to both heads.

        Value head    : Linear → ReLU → Linear(1)
                        Estimates V(s) — the scalar value of the state.

        Advantage head: Linear → ReLU → Linear(n_actions)
                        Estimates A(s, a) — the relative advantage of each action.

    Aggregation (identifiable form from the paper):

        Q(s, a) = V(s) + A(s, a) - (1/|A|) Σ_a' A(s, a')

    Subtracting the mean advantage rather than the max is preferred in
    practice because it is more stable the mean changes smoothly whereas
    the max can switch discontinuously during training

    """

    def __init__(self, obs_size: int, n_actions: int,
                 hidden_size: int = 128, advantage_hidden_size: int = 128):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, advantage_hidden_size),
            nn.ReLU(),
            nn.Linear(advantage_hidden_size, 1),
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_size, advantage_hidden_size),
            nn.ReLU(),
            nn.Linear(advantage_hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning Q-values via the dueling aggregation

        Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

        """
        features = self.trunk(x)                          # (B, hidden)
        V        = self.value_head(features)              # (B, 1)
        A        = self.advantage_head(features)          # (B, n_actions)

        Q = V + A - A.mean(dim=1, keepdim=True)          # (B, n_actions)
        return Q

    def value_advantage(self, x: torch.Tensor):
        """
        debugging info
        """
        features = self.trunk(x)
        V        = self.value_head(features)
        A        = self.advantage_head(features)
        return V, A


class DuelingAgent:
    """
    Dueling Deep Q-Network agent 
    """

    def __init__(self, env, cfg: DuelingConfig):
        self.cfg       = cfg
        self.device    = torch.device(cfg.device)

        obs_size       = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.policy_net = DuelingMlp(
            obs_size, self.n_actions,
            cfg.hidden_size, cfg.advantage_hidden_size
        ).to(self.device)

        self.target_net = DuelingMlp(
            obs_size, self.n_actions,
            cfg.hidden_size, cfg.advantage_hidden_size
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.memory     = ReplayBuffer(cfg.replay_capacity)
        self.steps_done = 0


    def epsilon(self) -> float:
        """Current ε for ε-greedy"""
        progress = min(self.steps_done / self.cfg.eps_decay_frames, 1.0)
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * (1.0 - progress)

    def select_action(self, state: np.ndarray, env) -> int:
        """
        ε-greedy action selection
        """
        if random.random() < self.epsilon():
            return env.action_space.sample()
        state_t = torch.as_tensor(state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(state_t).argmax(dim=1).item())


    def optimize_step(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient step
        """
        if len(self.memory) < self.cfg.min_replay_size:
            return None

        transitions = self.memory.sample(self.cfg.batch_size)
        batch       = Transition(*zip(*transitions))

        states      = torch.as_tensor(np.array(batch.state),      dtype=torch.float32, device=self.device)
        actions     = torch.tensor(batch.action,                   dtype=torch.long,    device=self.device).unsqueeze(1)
        rewards     = torch.tensor(batch.reward,                   dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones       = torch.tensor(batch.done,                     dtype=torch.float32, device=self.device)

        # Q(s, a) from dueling policy network — identical call to vanilla DQN
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # TD target from frozen dueling target network
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


    def value_advantage(self, state: np.ndarray):
        
        state_t = torch.as_tensor(state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        with torch.no_grad():
            V, A = self.policy_net.value_advantage(state_t)
        return float(V.item()), A.squeeze(0).cpu().tolist()

    # ── Persistence ───────────────────────────────────────────────────────────

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

    # ── Evaluation ────────────────────────────────────────────────────────────

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
                state_t = torch.as_tensor(state, dtype=torch.float32,
                                          device=self.device).unsqueeze(0)
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
    episode_rewards: list
    solved_frame:    Optional[int]
    total_frames:    int
    total_episodes:  int
    elapsed_seconds: float
    agent:           DuelingAgent



def train(cfg: Optional[DuelingConfig] = None) -> TrainingResults:
    """
    Run the full Dueling DQN training loop and return a TrainingResults object
    """
    import gymnasium as gym

    if cfg is None:
        cfg = DuelingConfig()

    print(f"  Dueling DQN — CartPole")    
    print(f"  Trunk hidden  : {cfg.hidden_size}")
    print(f"  Head hidden   : {cfg.advantage_hidden_size}  (value + advantage)")
    print(f"  Replay buffer : {cfg.replay_capacity:,}")
    print(f"  Batch size    : {cfg.batch_size}")
    print(f"  Max frames    : {cfg.max_frames:,}")
    print("\n")

    render_mode = "human" if cfg.render else None
    env         = gym.make(cfg.env_id, render_mode=render_mode)
    agent       = DuelingAgent(env, cfg)

    episode_rewards: list[float] = []
    frame_log:       list[int]   = []
    score_log:       list[float] = []

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
            fps     = frame / max(time.time() - t_start, 1e-6)
            elapsed = time.time() - t_start

            frame_log.append(frame)
            score_log.append(mean_r)

            bar_len  = 20
            progress = int(bar_len * frame / cfg.max_frames)
            bar      = "█" * progress + "░" * (bar_len - progress)

            print(
                f"[{bar}] {frame:>6,}/{cfg.max_frames:,}  "
                f"ep={ep_count:>4}  "
                f"mean_R={mean_r:>6.1f}  "
                f"eps={agent.epsilon():.3f}  "
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
        episode_rewards = episode_rewards,
        solved_frame    = solved_frame,
        total_frames    = frame,
        total_episodes  = ep_count,
        elapsed_seconds = elapsed,
        agent           = agent,
    )



def plot_results(results: TrainingResults) -> None:
    """
    Plot the score-vs-frame learning curve for a Dueling DQN run
    """
    import matplotlib.pyplot as plt

    cfg = results.agent.cfg
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(results.frame_log, results.score_log,
               s=12, alpha=0.35, color="royalblue", zorder=2,
               label="Mean score (100 ep window)")

    if len(results.score_log) >= 5:
        k        = min(5, len(results.score_log))
        smoothed = np.convolve(results.score_log, np.ones(k) / k, mode="valid")
        ax.plot(results.frame_log[k - 1:], smoothed,
                color="royalblue", linewidth=2.5, zorder=3,
                label=f"Smoothed ({k}-point moving avg)")


    if results.solved_frame is not None:
        ax.axvline(results.solved_frame, color="limegreen", linestyle=":",
                   linewidth=1.5, zorder=1,
                   label=f"Solved at frame {results.solved_frame:,}")

    ax.set_xlabel("Environment frames", fontsize=12)
    ax.set_ylabel("Mean episode score (last 100 ep)", fontsize=12)
    ax.set_title(f"Dueling DQN on {cfg.env_id} — Score vs Frames", fontsize=13)
    ax.set_xlim(0, cfg.max_frames)
    ax.set_ylim(0, 520)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
