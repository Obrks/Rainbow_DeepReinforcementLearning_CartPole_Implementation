"""
DQN with Multi-Step Returns for CartPole
Based on Sutton (1988) / Rainbow (Hessel et al. 2018).

 changes
  Simple DQN bootstraps from a single step:

      y = r_t + γ · max_a' Q(s_{t+1}, a')

  Multi-step DQN accumulates n real rewards before bootstrapping:

      G_t^(n) = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{n-1}·r_{t+n-1}
      y       = G_t^(n) + γ^n · max_a' Q(s_{t+n}, a')

  This propagates reward signals faster through the Q-network instead of
  waiting n gradient steps for a reward to influence states n steps back,
  it happens in a single update

  One thing changes in the architecture:

    An n-step buffer (NStepBuffer) accumulates the last n transitions
    When it is full, it computes G_t^(n) and emits a single combined
    transition (s_t, a_t, G_t^(n), s_{t+n}, done) into the replay buffer

  parameter:
      n_steps (n): number of steps to accumulate before bootstrapping.
                   n=1 →   Simple DQN bootstraps from a single step:
    DQN, n=3 is the Rainbow default.
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
class MultiStepConfig:
    """
    parameters for the Multi-Step DQN agent
    """
    #Network  
    learning_rate:      float = 1e-3
    hidden_size:        int   = 128

    # Replay buffer
    replay_capacity:    int   = 50_000
    min_replay_size:    int   = 1_000
    batch_size:         int   = 64

    #Multi-step
    n_steps:            int   = 3       # number of steps to accumulate
                                        # Rainbow default is 3

    # RL 
    gamma:              float = 0.99
    target_update:      int   = 1_000
    train_freq:         int   = 1

    #Exploration
    eps_start:          float = 1.0
    eps_end:            float = 0.01
    eps_decay_frames:   int   = 10_000

    # Training loop
    max_frames:         int   = 200_000
    log_interval:       int   = 5_000
    solve_threshold:    float = 475.0

    # Misc 
    render:             bool  = False
    device:             str   = "cpu"
    env_id:             str   = "CartPole-v1"

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """
    Circular experience replay buffer with uniform random sampling
    """

    def __init__(self, capacity: int):
        self.memory: deque = deque(maxlen=capacity)

    def push(self, state, action, reward: float, next_state, done: float) -> None:
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

class NStepBuffer:
    """
    Accumulates the last n transitions and emits a single n-step transition

    For each step t, it holds transitions (s_t, a_t, r_t), ..., (s_{t+n-1}, ...).
    When the buffer is full or the episode ends, it computes the n-step return:

        G_t^(n) = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^{n-1}·r_{t+n-1}

    and emits the combined transition:

        (s_t, a_t, G_t^(n), s_{t+n}, done)
    """

    def __init__(self, n_steps: int, gamma: float):
        self.n_steps = n_steps
        self.gamma   = gamma
        self.buffer  = deque(maxlen=n_steps)   # holds raw (s, a, r, s', done)

    def push(self, state, action, reward: float,
             next_state, done: float) -> Optional[tuple]:
        """
        Add one transition
        """
        self.buffer.append((state, action, reward, next_state, done))

        # Not enough steps yet and episode hasn't ended — wait
        if len(self.buffer) < self.n_steps and not done:
            return None

        # Compute G_t^(n) = Σ_{k=0}^{n-1} γ^k · r_{t+k}
        G        = 0.0
        discount = 1.0
        for (_, _, r, _, _) in self.buffer:
            G        += discount * r
            discount *= self.gamma

        # s_t, a_t come from the oldest entry; s_{t+n} from the newest
        s_t,  a_t,  _,  _,   _    = self.buffer[0]
        _,    _,    _,  s_tn, done = self.buffer[-1]

        # Clear buffer on episode end so it does not bleed into the next episode
        if done:
            self.buffer.clear()

        return (s_t, a_t, G, s_tn, done)

    def flush(self) -> list[tuple]:
        """
        Emit all remaining partial n-step transitions at episode end
        """
        transitions = []
        while self.buffer:
            G        = 0.0
            discount = 1.0
            for (_, _, r, _, _) in self.buffer:
                G        += discount * r
                discount *= self.gamma
            s_t, a_t, _, _,   _    = self.buffer[0]
            _,   _,   _, s_tn, done = self.buffer[-1]
            transitions.append((s_t, a_t, G, s_tn, done))
            self.buffer.popleft()
        return transitions

    def __len__(self) -> int:
        return len(self.buffer)


class MultiStepMlp(nn.Module):
    """
    Three-layer MLP Q-network same as simple DQN
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


class MultiStepAgent:
    """
    DQN agent with n-step returns
    """

    def __init__(self, env, cfg: MultiStepConfig):
        self.cfg       = cfg
        self.device    = torch.device(cfg.device)

        obs_size       = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.policy_net = MultiStepMlp(obs_size, self.n_actions, cfg.hidden_size).to(self.device)
        self.target_net = MultiStepMlp(obs_size, self.n_actions, cfg.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer   = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.memory      = ReplayBuffer(cfg.replay_capacity)
        self.nstep_buf   = NStepBuffer(cfg.n_steps, cfg.gamma)
        self.steps_done  = 0

        # γ^n used in the TD target 
        self.gamma_n = cfg.gamma ** cfg.n_steps


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
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.policy_net(state_t).argmax(dim=1).item())

    def store(self, state, action, reward: float,
              next_state, done: float) -> None:
        """
        Pass a raw transition through the n-step buffer into the replay buffer
        """
        transition = self.nstep_buf.push(state, action, reward, next_state, done)
        if transition is not None:
            self.memory.push(*transition)

    def flush_episode(self) -> None:
        """
        Drain the n-step buffer at episode end
        """
        for transition in self.nstep_buf.flush():
            self.memory.push(*transition)


    def optimize_step(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one Multi-Step DQN gradient step
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

        # Q(s_t, a_t) from policy network
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # reward here is already G_t^(n) bootstrap discount is γ^n not 
        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1).values
            targets = rewards + self.gamma_n * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def save(self, path: str) -> None:
        """
        Save the policy network weights and training 
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
    episode_rewards: list
    solved_frame:    Optional[int]
    total_frames:    int
    total_episodes:  int
    elapsed_seconds: float
    agent:           MultiStepAgent

def train(cfg: Optional[MultiStepConfig] = None) -> TrainingResults:
    """
    Run the full Multi-Step DQN training loop and return a TrainingResults object
    """
    import gymnasium as gym

    if cfg is None:
        cfg = MultiStepConfig()

    print("MultiStep DQN — CartPole")
    print(f"  Replay buffer : {cfg.replay_capacity:,}")
    print(f"  Batch size    : {cfg.batch_size}")
    print(f"  Max frames    : {cfg.max_frames:,}")
    print("\n")

    render_mode = "human" if cfg.render else None
    env         = gym.make(cfg.env_id, render_mode=render_mode)
    agent       = MultiStepAgent(env, cfg)

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

        # Route through n-step buffer before replay buffer
        agent.store(state, action, float(reward), next_state, float(done))

        # Flush partial transitions at episode end
        if done:
            agent.flush_episode()

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
    import matplotlib.pyplot as plt

    cfg = results.agent.cfg
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(results.frame_log, results.score_log,
               s=12, alpha=0.35, color="crimson", zorder=2,
               label="Mean score (100 ep window)")

    if len(results.score_log) >= 5:
        k        = min(5, len(results.score_log))
        smoothed = np.convolve(results.score_log, np.ones(k) / k, mode="valid")
        ax.plot(results.frame_log[k - 1:], smoothed,
                color="crimson", linewidth=2.5, zorder=3,
                label=f"Smoothed ({k}-point moving avg)")

    if results.solved_frame is not None:
        ax.axvline(results.solved_frame, color="limegreen", linestyle=":",
                   linewidth=1.5, zorder=1,
                   label=f"Solved at frame {results.solved_frame:,}")

    ax.set_xlabel("Environment frames", fontsize=12)
    ax.set_ylabel("Mean episode score (last 100 ep)", fontsize=12)
    ax.set_title(f"Multi-Step DQN (n={cfg.n_steps}) on {cfg.env_id} — Score vs Frames",
                 fontsize=13)
    ax.set_xlim(0, cfg.max_frames)
    ax.set_ylim(0, 520)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
