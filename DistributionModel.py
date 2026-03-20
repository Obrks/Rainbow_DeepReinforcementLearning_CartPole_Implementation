"""
Distributional DQN for CartPole
Based on Bellemare, Dabney & Munos (2017) / Rainbow (Hessel et al. 2018)

changes
  Simple DQN learns a single scalar Q(s, a) = E[Z(s, a)] per action.
  Distributional DQN learns the full return distribution Z(s, a)
  represented as a fixed categorical distribution over N atoms:

      z_i = V_min + i · Δz,   i = 0, ..., N-1
      Δz  = (V_max - V_min) / (N - 1)

  The network outputs a probability vector p(s, a) ∈ Δ^N for each action,
  so the full output shape is (batch, n_actions, N_atoms) instead of
  (batch, n_actions)

  Action selection still uses the mean (expected) Q-value:

      Q(s, a) = Σ_i z_i · p_i(s, a)

  The target distribution is computed via the Bellman operator on atoms:

      T z_i = r + γ · z_i      

  and then projected back onto the fixed support via linear interpolatio
  The loss is the cross-entropy between the projected target
  distribution and the predicted distribution, rather than an MSE on scalars

  Three things 
    1. Network outputs N_atoms probabilities per action instead of 1 scalar
    2. Target is a projected categorical distribution, not a scalar y
    3. Loss is cross-entropy, not Huber loss

  parameters:
      n_atoms:  number of distribution support points (default 51, from Rainbow Paper)
      v_min:    minimum support value
      v_max:    maximum support value

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
class DistributionalConfig:
    """
    parameters for the Distributional agent and training loop
    """
    #Network 
    learning_rate:      float = 1e-3
    hidden_size:        int   = 128

    # Distributional
    n_atoms:            int   = 51      # number of support atoms
    v_min:              float = 0.0     # minimum support value
    v_max:              float = 500.0   # maximum support value (CartPole max)

    # Replay buffer
    replay_capacity:    int   = 50_000
    min_replay_size:    int   = 1_000
    batch_size:         int   = 64

    #RL
    gamma:              float = 0.99
    target_update:      int   = 1_000
    train_freq:         int   = 1

    # Exploration
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
    """Circular replay buffer with uniform random samplin"""

    def __init__(self, capacity: int):
        self.memory: deque = deque(maxlen=capacity)

    def push(self, state, action, reward: float, next_state, done: float) -> None:
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DistributionalMlp(nn.Module):
    """
    Three-layer MLP that outputs a categorical return distribution per action

    Unlike simple DQN which outputs one scalar Q-value per action, Distributional outputs
    N_atoms probability values per action, one probability for each atom in the
    support [V_min, V_max]. The output is passed through softmax to form a valid
    probability distribution
    """

    def __init__(self, obs_size: int, n_actions: int,
                 n_atoms: int, hidden_size: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms   = n_atoms

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions * n_atoms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a softmax probability distribution
        """
        logits = self.net(x)                                    # (B, n_actions * n_atoms)
        logits = logits.view(-1, self.n_actions, self.n_atoms)  # (B, n_actions, n_atoms)
        return F.softmax(logits, dim=-1)                        # probability distribution


class DistributionalAgent:
    """
    Distributional DQN agent 

    Instead of learning E[Z(s,a)], the agent learns the full distribution
    Z(s,a) over returns, represented as a categorical distribution over a
    fixed support of N atoms equally spaced between V_min and V_max

    Key differences from DQNAgent:
      1. Network outputs (n_actions, n_atoms) probabilities per state.
      2. Action selection uses expected Q-values: Q(s,a) = Σ z_i · p_i(s,a).
      3. optimize_step() uses the distributional Bellman operator:
           - Project the Bellman-updated atoms T·z_i = r + γ·z_i onto the
             fixed support via linear interpolation (categorical projection).
           - Minimise the cross-entropy between the projected target
             distribution and the predicted distribution.
      4. No scalar TD error, loss cross-entropy.

    Everything else (replay buffer, target network, epsilon-greedy,
    gradient clipping) is identical to vanilla DQN
    """

    def __init__(self, env, cfg: DistributionalConfig):
        self.cfg       = cfg
        self.device    = torch.device(cfg.device)

        obs_size       = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.policy_net = DistributionalMlp(obs_size, self.n_actions,
                                 cfg.n_atoms, cfg.hidden_size).to(self.device)
        self.target_net = DistributionalMlp(obs_size, self.n_actions,
                                 cfg.n_atoms, cfg.hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.memory     = ReplayBuffer(cfg.replay_capacity)
        self.steps_done = 0

        # ── Fixed atom support: z_i = V_min + i·Δz ───────────────────────────
        self.atoms = torch.linspace(
            cfg.v_min, cfg.v_max, cfg.n_atoms, device=self.device
        )                                                    # shape: (N,)
        self.delta_z = (cfg.v_max - cfg.v_min) / (cfg.n_atoms - 1)


    def epsilon(self) -> float:
        """Current ε for ε-greedy"""
        progress = min(self.steps_done / self.cfg.eps_decay_frames, 1.0)
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * (1.0 - progress)

    def select_action(self, state: np.ndarray, env) -> int:
        """
        ε-greedy action selection using expected Q-values

        The expected Q-value for each action is computed as:
            Q(s, a) = Σ_i z_i · p_i(s, a)

        where z_i are the fixed atom values and p_i(s, a) are the predicted
        probabilities. This collapses the distribution to a scalar for
        action selection, identical in interface to vanilla DQN
        """
        if random.random() < self.epsilon():
            return env.action_space.sample()

        state_t = torch.as_tensor(state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist    = self.policy_net(state_t)               # (1, n_actions, n_atoms)
            q_vals  = (dist * self.atoms).sum(-1)            # (1, n_actions)
        return int(q_vals.argmax(dim=1).item())


    def _project_distribution(self, next_dist: torch.Tensor,
                               rewards: torch.Tensor,
                               dones: torch.Tensor) -> torch.Tensor:
        """
        Project the Bellman-updated distribution onto the fixed atom support

        For each atom z_i in the support, compute the shifted atom:

            T z_i = r + γ · z_i     

        Then distribute the probability p_i onto the two neighbouring atoms
        via linear interpolation (the "categorical projection" algorithm from
        the Distributional paper, Algorithm 1):

            lower = floor((T z_i - V_min) / Δz)
            upper = ceil( (T z_i - V_min) / Δz)

            m_lower += p_i · (upper - b)   where b = (T z_i - V_min) / Δz
            m_upper += p_i · (b - lower)
        """
        B     = rewards.shape[0]
        N     = self.cfg.n_atoms
        v_min = self.cfg.v_min
        v_max = self.cfg.v_max

        # T z_i = r + γ · z_i, clipped to [V_min, V_max]
        # shape: (batch, n_atoms)
        atoms_expanded = self.atoms.unsqueeze(0).expand(B, -1)          # (B, N)
        r_expanded     = rewards.unsqueeze(1).expand_as(atoms_expanded) # (B, N)
        d_expanded     = dones.unsqueeze(1).expand_as(atoms_expanded)   # (B, N)

        tz = r_expanded + self.cfg.gamma * atoms_expanded * (1.0 - d_expanded)
        tz = tz.clamp(v_min, v_max)                                     # (B, N)

        # Fractional position of each T z_i on the support grid
        b = (tz - v_min) / self.delta_z                                 # (B, N)

        lower = b.floor().long().clamp(0, N - 1)                        # (B, N)
        upper = b.ceil().long().clamp(0, N - 1)                         # (B, N)

        # Distribute probability mass to neighbouring atoms
        m = torch.zeros(B, N, device=self.device)                       # (B, N)

        # Fraction going to upper neighbour: (b - lower)
        # Fraction going to lower neighbour: (upper - b)
        upper_frac = (b - lower.float())                                 # (B, N)
        lower_frac = (upper.float() - b)                                 # (B, N)

        # scatter_add accumulates contributions at each atom index
        m.scatter_add_(1, lower, next_dist * lower_frac)
        m.scatter_add_(1, upper, next_dist * upper_frac)

        return m   # projected target distribution, shape (B, N)

    def optimize_step(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one Distributional gradient step
        """
        if len(self.memory) < self.cfg.min_replay_size:
            return None

        transitions = self.memory.sample(self.cfg.batch_size)
        batch       = Transition(*zip(*transitions))

        states      = torch.as_tensor(np.array(batch.state),      dtype=torch.float32, device=self.device)
        actions     = torch.tensor(batch.action,                   dtype=torch.long,    device=self.device)
        rewards     = torch.tensor(batch.reward,                   dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones       = torch.tensor(batch.done,                     dtype=torch.float32, device=self.device)

        # Target distribution 
        with torch.no_grad():
            # Full distribution for all actions in next states
            next_dist_all = self.target_net(next_states)          # (B, A, N)

            # Greedy action selection via expected Q-values
            next_q        = (next_dist_all * self.atoms).sum(-1)  # (B, A)
            next_actions  = next_q.argmax(dim=1)                  # (B,)

            # Distribution for the greedy action only
            idx       = next_actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.cfg.n_atoms)
            next_dist = next_dist_all.gather(1, idx).squeeze(1)   # (B, N)

            # Project onto fixed support via categorical projection
            target_dist = self._project_distribution(next_dist, rewards, dones)  # (B, N)

        # Predicted distribution for taken actions 
        dist_all = self.policy_net(states)                         # (B, A, N)
        idx      = actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.cfg.n_atoms)
        dist     = dist_all.gather(1, idx).squeeze(1)             # (B, N)

        # Cross-entropy 
        # Add small epsilon inside log for numerical stability
        loss = -(target_dist * dist.clamp(min=1e-8).log()).sum(-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def sync_target(self) -> None:
        """Copy policy network weights into the target network"""
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
                state_t = torch.as_tensor(state, dtype=torch.float32,
                                          device=self.device).unsqueeze(0)
                with torch.no_grad():
                    dist   = self.policy_net(state_t)         # (1, A, N)
                    q_vals = (dist * self.atoms).sum(-1)      # (1, A)
                    action = int(q_vals.argmax(dim=1).item())
                state, reward, term, trunc, _ = env.step(action)
                total += reward
                done   = term or trunc
            scores.append(total)
        return float(np.mean(scores))


    def get_distribution(self, state: np.ndarray) -> np.ndarray:
        """
        Return the predicted return distribution for all actions in a state
        """
        state_t = torch.as_tensor(state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.policy_net(state_t).squeeze(0)   # (n_actions, n_atoms)
        return dist.cpu().numpy()


@dataclass
class TrainingResults:
    """
    data from the training 
    """
    frame_log:       list
    score_log:       list
    episode_rewards: list
    solved_frame:    Optional[int]
    total_frames:    int
    total_episodes:  int
    elapsed_seconds: float
    agent:           DistributionalAgent

def train(cfg: Optional[DistributionalConfig] = None) -> TrainingResults:
    """
    Run the full Distributional training loop and return a TrainingResults object
    """
    import gymnasium as gym

    if cfg is None:
        cfg = DistributionalConfig()

    delta_z = (cfg.v_max - cfg.v_min) / (cfg.n_atoms - 1)

    print(f"  Distributional DQN — CartPole")
    print(f"  Atoms         : {cfg.n_atoms}  ({cfg.v_min} to {cfg.v_max}, Δz={delta_z:.2f})")
    print(f"  Replay buffer : {cfg.replay_capacity:,}")
    print(f"  Batch size    : {cfg.batch_size}")
    print(f"  Max frames    : {cfg.max_frames:,}")
    print("\n")

    render_mode = "human" if cfg.render else None
    env         = gym.make(cfg.env_id, render_mode=render_mode)
    agent       = DistributionalAgent(env, cfg)

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

def plot_results(results: TrainingResults, save_path: str = "Distributional_rewards.png") -> None:
    """
    Plot score-vs-frame learning curve for a Distributional run
    """
    import matplotlib.pyplot as plt

    cfg = results.agent.cfg
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(results.frame_log, results.score_log,
               s=12, alpha=0.35, color="teal", zorder=2,
               label="Mean score (100 ep window)")

    if len(results.score_log) >= 5:
        k        = min(5, len(results.score_log))
        smoothed = np.convolve(results.score_log, np.ones(k) / k, mode="valid")
        ax.plot(results.frame_log[k - 1:], smoothed,
                color="teal", linewidth=2.5, zorder=3,
                label=f"Smoothed ({k}-point moving avg)")

    if results.solved_frame is not None:
        ax.axvline(results.solved_frame, color="limegreen", linestyle=":",
                   linewidth=1.5, zorder=1,
                   label=f"Solved at frame {results.solved_frame:,}")

    ax.set_xlabel("Environment frames", fontsize=12)
    ax.set_ylabel("Mean episode score (last 100 ep)", fontsize=12)
    ax.set_title(
        f"Distributional (N={cfg.n_atoms}, [{cfg.v_min}, {cfg.v_max}]) on {cfg.env_id}",
        fontsize=13
    )
    ax.set_xlim(0, cfg.max_frames)
    ax.set_ylim(0, 520)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved → {save_path}")


def plot_distribution(agent: DistributionalAgent, state: np.ndarray,
                      action_labels: Optional[list] = None) -> None:
    """
    Plot the predicted return distribution for each action in a given state
    """
    import matplotlib.pyplot as plt

    dist  = agent.get_distribution(state)          # (n_actions, n_atoms)
    atoms = agent.atoms.cpu().numpy()
    cfg   = agent.cfg

    if action_labels is None:
        action_labels = [f"Action {i}" for i in range(agent.n_actions)]

    colours = ["steelblue", "darkorange", "mediumseagreen", "mediumpurple"]
    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (label, colour) in enumerate(zip(action_labels, colours)):
        q_val = float((dist[i] * atoms).sum())
        ax.bar(atoms, dist[i], width=(cfg.v_max - cfg.v_min) / cfg.n_atoms * 0.8,
               alpha=0.6, color=colour, label=f"{label}  (E[Z] = {q_val:.1f})")

    ax.set_xlabel("Return", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("Distributional — Return Distribution per Action", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
