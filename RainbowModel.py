"""
Rainbow DQN (without Dueling) for CartPole
Based on Hessel et al. (2018) — "Rainbow: Combining Improvements in Deep RL"

included
    Double DQN          
    Prioritized Replay  
    Noisy Nets          
    Multi-step Returns  
    Distributional RL   
    Dueling Networks    — excluded because not effective enough for CartPole

  1. The network outputs a distribution (n_actions, n_atoms)
     using NoisyLinear layers throughout combining Noisy Nets + Distributional RL

  2. Transitions are routed through an NStepBuffer before the replay buffer,
     so the stored reward is G_t^(n) Multi-step return

  3. The replay buffer is a PrioritizedReplayBuffer, PER

  4. The TD target uses Double DQN decoupling:
       a* = argmax_a' E[Z_policy(s', a')]     (policy net selects)
       y  = project(G^(n) + γ^n · Z_target(s', a*))  (target net evaluates)

  5. The loss is IS-weighted cross-entropy

  6. noise is reset at every gradient step 
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
class RainbowConfig:
    """
    parameters for the Rainbow agent 
    Combines parameters from all five included components
    """
    #Network 
    learning_rate:      float = 1e-3
    hidden_size:        int   = 128

    # Noisy Nets 
    sigma_init:         float = 0.5     # initial σ for NoisyLinear layers

    # Distributional 
    n_atoms:            int   = 51      # number of support atoms
    v_min:              float = 0.0     # minimum support value
    v_max:              float = 500.0   # maximum support value (CartPole max)

    #  Multi-step
    n_steps:            int   = 3       # steps to accumulate before bootstrap

    # Prioritized replay
    replay_capacity:    int   = 50_000
    min_replay_size:    int   = 1_000
    batch_size:         int   = 64
    alpha:              float = 0.6     # prioritization exponent
    beta_start:         float = 0.4     # initial IS exponent (annealed → 1.0)
    per_eps:            float = 1e-5    # priority floor

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




class NoisyLinear(nn.Module):
    """
    Linear layer with factorised Gaussian noise

        y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.empty(out_features))

        bound = 1.0 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(out_features))

        self.reset_noise()

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Resample factorised Gaussian noise"""
        eps_i = self._f(torch.randn(self.in_features,  device=self.weight_mu.device))
        eps_j = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(eps_j.outer(eps_i))
        self.bias_epsilon.copy_(eps_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        return F.linear(x, weight, bias)


class RainbowMlp(nn.Module):
    """
    MLP Q-network combining Noisy Nets and Distributional RL 

    All linear layers use NoisyLinear so there is no ε-greedy
    Output shape is (batch, n_actions, n_atoms) a probability distribution
    over the return support for each action
    """

    def __init__(self, obs_size: int, n_actions: int, n_atoms: int,
                 hidden_size: int = 128, sigma_init: float = 0.5):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms   = n_atoms

        self.fc1 = NoisyLinear(obs_size,              hidden_size, sigma_init)
        self.fc2 = NoisyLinear(hidden_size,            hidden_size, sigma_init)
        self.fc3 = NoisyLinear(hidden_size, n_actions * n_atoms,   sigma_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax distribution, shape (batch, n_actions, n_atoms)"""
        x      = F.relu(self.fc1(x))
        x      = F.relu(self.fc2(x))
        logits = self.fc3(x)                                    # (B, A*N)
        logits = logits.view(-1, self.n_actions, self.n_atoms)  # (B, A, N)
        return F.softmax(logits, dim=-1)

    def reset_noise(self) -> None:
        """Resample noise in all three NoisyLinear layers"""
        self.fc1.reset_noise()
        self.fc2.reset_noise()
        self.fc3.reset_noise()


class SumTree:
    """Binary sum-tree for PER"""

    def __init__(self, capacity: int):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data      = np.empty(capacity, dtype=object)
        self.write     = 0
        self.n_entries = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, value: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        return self._retrieve(right, value - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        leaf_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(leaf_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        delta               = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, delta)

    def sample(self, batch_size: int):
        indices    = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float64)
        data       = []
        segment    = self.total / batch_size
        for i in range(batch_size):
            val = random.uniform(segment * i, segment * (i + 1))
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
    Replay buffer with prioritized sampling
    Sampling probability: P(i) = p_i^α / Σ p_j^α
    IS correction weight: w_i = (N · P(i))^{-β} / max_j w_j
    β is annealed from beta_start → 1.0 over training
    """

    def __init__(self, capacity: int, alpha: float,
                 beta_start: float, per_eps: float):
        self.tree     = SumTree(capacity)
        self.alpha    = alpha
        self.per_eps  = per_eps
        self.max_prio = 1.0

    def push(self, state, action, reward: float, next_state, done: float) -> None:
        self.tree.add(self.max_prio ** self.alpha,
                      Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int, beta: float):
        indices, priorities, transitions = self.tree.sample(batch_size)
        N       = len(self.tree)
        probs   = priorities / self.tree.total
        weights = (N * probs) ** (-beta)
        weights /= weights.max()
        return indices, weights.astype(np.float32), transitions

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            prio          = (abs(float(err)) + self.per_eps) ** self.alpha
            self.max_prio = max(self.max_prio, prio)
            self.tree.update(int(idx), prio)

    def __len__(self) -> int:
        return len(self.tree)


class NStepBuffer:
    """
    Accumulates n raw transitions and emits one n-step transition

    """

    def __init__(self, n_steps: int, gamma: float):
        self.n_steps = n_steps
        self.gamma   = gamma
        self.buffer  = deque(maxlen=n_steps)

    def push(self, state, action, reward: float,
             next_state, done: float) -> Optional[tuple]:
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) < self.n_steps and not done:
            return None
        G = sum(self.gamma ** k * r for k, (_, _, r, _, _) in enumerate(self.buffer))
        s_t,  a_t, _, _,   _    = self.buffer[0]
        _,    _,   _, s_tn, done = self.buffer[-1]
        if done:
            self.buffer.clear()
        return (s_t, a_t, G, s_tn, done)

    def flush(self) -> list[tuple]:
        transitions = []
        while self.buffer:
            G = sum(self.gamma ** k * r for k, (_, _, r, _, _) in enumerate(self.buffer))
            s_t, a_t, _, _,   _    = self.buffer[0]
            _,   _,   _, s_tn, done = self.buffer[-1]
            transitions.append((s_t, a_t, G, s_tn, done))
            self.buffer.popleft()
        return transitions

    def __len__(self) -> int:
        return len(self.buffer)


class RainbowAgent:
    """
    Rainbow agent combining five DQN improvements

    Component interactions in optimize_step():
      
        1. Sample batch from PrioritizedReplayBuffer  (PER)        
        2. Reset noise in policy + target nets        (Noisy Nets) 
        3. Compute n-step TD target:                               
             a* = argmax_a E[Z_policy(s', a')]        (Double DQN) 
             target = project(G^n + γ^n·Z_target(s',a*))     
        4. IS-weighted cross-entropy loss             
        5. Update priorities with fresh TD errors     (PER)        
      
    """

    def __init__(self, env, cfg: RainbowConfig):
        self.cfg       = cfg
        self.device    = torch.device(cfg.device)

        obs_size       = env.observation_space.shape[0]
        self.n_actions = env.action_space.n

        self.policy_net = RainbowMlp(
            obs_size, self.n_actions, cfg.n_atoms, cfg.hidden_size, cfg.sigma_init
        ).to(self.device)
        self.target_net = RainbowMlp(
            obs_size, self.n_actions, cfg.n_atoms, cfg.hidden_size, cfg.sigma_init
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.memory     = PrioritizedReplayBuffer(
            cfg.replay_capacity, cfg.alpha, cfg.beta_start, cfg.per_eps
        )
        self.nstep_buf  = NStepBuffer(cfg.n_steps, cfg.gamma)
        self.steps_done = 0

        # Fixed atom support and discount
        self.atoms   = torch.linspace(cfg.v_min, cfg.v_max,
                                      cfg.n_atoms, device=self.device)
        self.delta_z = (cfg.v_max - cfg.v_min) / (cfg.n_atoms - 1)
        self.gamma_n = cfg.gamma ** cfg.n_steps


    def beta(self) -> float:
        """β annealed from beta_start → 1.0 over max_frames."""
        progress = min(self.steps_done / self.cfg.max_frames, 1.0)
        return self.cfg.beta_start + (1.0 - self.cfg.beta_start) * progress


    def select_action(self, state: np.ndarray) -> int:
        """
        Greedy action selection via expected Q-values E[Z(s,a)] = Σ z_i·p_i.
        No epsilon — exploration comes from the NoisyLinear layers.

        Args:
            state: current environment observation.

        Returns:
            Integer action index.
        """
        state_t = torch.as_tensor(state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist   = self.policy_net(state_t)          # (1, A, N)
            q_vals = (dist * self.atoms).sum(-1)       # (1, A)
        return int(q_vals.argmax(dim=1).item())


    def store(self, state, action, reward: float,
              next_state, done: float) -> None:
        """Route raw transition through n-step buffer into replay buffer."""
        transition = self.nstep_buf.push(state, action, reward, next_state, done)
        if transition is not None:
            self.memory.push(*transition)

    def flush_episode(self) -> None:
        """Drain partial transitions from n-step buffer at episode end."""
        for t in self.nstep_buf.flush():
            self.memory.push(*t)


    def _project_distribution(self, next_dist: torch.Tensor,
                               rewards: torch.Tensor,
                               dones: torch.Tensor) -> torch.Tensor:
        """
        Project Bellman-shifted distribution onto the fixed atom support.

        T z_i = G^(n) + γ^n · z_i  clipped to [V_min, V_max]
        Mass of each T z_i is split between its two neighbouring grid atoms
        via linear interpolation.

        Args:
            next_dist: shape (batch, n_atoms) — target distribution for best action.
            rewards:   shape (batch,)         — n-step return G^(n).
            dones:     shape (batch,)         — 1.0 if episode ended.

        Returns:
            Projected target distribution, shape (batch, n_atoms).
        """
        B, N  = rewards.shape[0], self.cfg.n_atoms
        v_min, v_max = self.cfg.v_min, self.cfg.v_max

        atoms_exp = self.atoms.unsqueeze(0).expand(B, -1)
        r_exp     = rewards.unsqueeze(1).expand_as(atoms_exp)
        d_exp     = dones.unsqueeze(1).expand_as(atoms_exp)

        # Shifted atoms using n-step discount γ^n
        tz = (r_exp + self.gamma_n * atoms_exp * (1.0 - d_exp)).clamp(v_min, v_max)
        b  = (tz - v_min) / self.delta_z

        lower = b.floor().long().clamp(0, N - 1)
        upper = b.ceil().long().clamp(0, N - 1)

        m = torch.zeros(B, N, device=self.device)
        m.scatter_add_(1, lower, next_dist * (upper.float() - b))
        m.scatter_add_(1, upper, next_dist * (b - lower.float()))
        return m


    def optimize_step(self) -> Optional[float]:
        """
        One Rainbow gradient step 
        """
        if len(self.memory) < self.cfg.min_replay_size:
            return None

        # Prioritized sampling 
        beta                          = self.beta()
        indices, weights, transitions = self.memory.sample(self.cfg.batch_size, beta)
        batch                         = Transition(*zip(*transitions))

        states      = torch.as_tensor(np.array(batch.state),      dtype=torch.float32, device=self.device)
        actions     = torch.tensor(batch.action,                   dtype=torch.long,    device=self.device)
        rewards     = torch.tensor(batch.reward,                   dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones       = torch.tensor(batch.done,                     dtype=torch.float32, device=self.device)
        weights_t   = torch.as_tensor(weights,                     dtype=torch.float32, device=self.device)

        #Resample noise 
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # Double DQN  policy net selects best next action
        with torch.no_grad():
            # action with highest expected return in s
            next_dist_policy = self.policy_net(next_states)          # (B, A, N)
            next_q_policy    = (next_dist_policy * self.atoms).sum(-1)  # (B, A)
            best_actions     = next_q_policy.argmax(dim=1)           # (B,)

            # Target net evaluates that action's distribution 
            next_dist_all = self.target_net(next_states)             # (B, A, N)
            idx           = best_actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.cfg.n_atoms)
            next_dist     = next_dist_all.gather(1, idx).squeeze(1)  # (B, N)

            target_dist = self._project_distribution(next_dist, rewards, dones)

        #IS-weighted cross-entropy loss 
        dist_all  = self.policy_net(states)                          # (B, A, N)
        idx       = actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.cfg.n_atoms)
        dist      = dist_all.gather(1, idx).squeeze(1)              # (B, N)

        cross_entropy = -(target_dist * dist.clamp(min=1e-8).log()).sum(-1)  # (B,)
        loss          = (weights_t * cross_entropy).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        with torch.no_grad():
            td_errors = cross_entropy.detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        return loss.item()

    def sync_target(self) -> None:
        """Copy policy network weights into the target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def mean_noise_magnitude(self) -> float:
        """Mean |σ| across all NoisyLinear layers tracking exploration decay"""
        sigmas = []
        for layer in [self.policy_net.fc1, self.policy_net.fc2, self.policy_net.fc3]:
            sigmas.append(layer.weight_sigma.abs().mean().item())
            sigmas.append(layer.bias_sigma.abs().mean().item())
        return float(np.mean(sigmas))


    def save(self, path: str) -> None:
        """Save policy weights and training"""
        torch.save({
            "policy_state":    self.policy_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "steps_done":      self.steps_done,
        }, path)
        print(f"Checkpoint saved → {path}")

    def load(self, path: str) -> None:
        """Load weights and training state from a checkpoint"""
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
                    dist   = self.policy_net(state_t)
                    q_vals = (dist * self.atoms).sum(-1)
                    action = int(q_vals.argmax(dim=1).item())
                state, reward, term, trunc, _ = env.step(action)
                total += reward
                done   = term or trunc
            scores.append(total)
        return float(np.mean(scores))


@dataclass
class TrainingResults:
    
    frame_log:       list
    score_log:       list
    noise_log:       list
    beta_log:        list
    episode_rewards: list
    solved_frame:    Optional[int]
    total_frames:    int
    total_episodes:  int
    elapsed_seconds: float
    agent:           RainbowAgent



def train(cfg: Optional[RainbowConfig] = None) -> TrainingResults:
    """
    Run the full Rainbow training loop and return a TrainingResults object
    """
    import gymnasium as gym

    if cfg is None:
        cfg = RainbowConfig()

    print(f"  Rainbow (with no Dueling) — CartPole")
    print(f"  Atoms         : {cfg.n_atoms}  [{cfg.v_min}, {cfg.v_max}]")
    print(f"  n-steps       : {cfg.n_steps}  (γ^n = {cfg.gamma**cfg.n_steps:.4f})")
    print(f"  PER α / β₀    : {cfg.alpha} / {cfg.beta_start}")
    print(f"  σ_init        : {cfg.sigma_init}")
    print(f"  Max frames    : {cfg.max_frames:,}")
    print("\n")

    render_mode = "human" if cfg.render else None
    env         = gym.make(cfg.env_id, render_mode=render_mode)
    agent       = RainbowAgent(env, cfg)

    episode_rewards: list[float] = []
    frame_log:       list[int]   = []
    score_log:       list[float] = []
    noise_log:       list[float] = []
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
        action                              = agent.select_action(state)
        next_state, reward, term, trunc, _ = env.step(action)
        done                               = term or trunc

        agent.store(state, action, float(reward), next_state, float(done))
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
            noise   = agent.mean_noise_magnitude()
            beta    = agent.beta()
            fps     = frame / max(time.time() - t_start, 1e-6)
            elapsed = time.time() - t_start

            frame_log.append(frame)
            score_log.append(mean_r)
            noise_log.append(noise)
            beta_log.append(beta)

            bar_len  = 20
            progress = int(bar_len * frame / cfg.max_frames)
            bar      = "█" * progress + "░" * (bar_len - progress)

            print(
                f"[{bar}] {frame:>6,}/{cfg.max_frames:,}  "
                f"ep={ep_count:>4}  "
                f"mean_R={mean_r:>6.1f}  "
                f"σ={noise:.4f}  "
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
        noise_log       = noise_log,
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
    Plot the score-vs-frame learning curve for a Rainbow run
    """
    import matplotlib.pyplot as plt
 
    cfg      = results.agent.cfg
    fig, ax  = plt.subplots(figsize=(10, 5))
 
    def _smooth(arr, k=5):
        if len(arr) < k:
            return arr
        return np.convolve(arr, np.ones(k) / k, mode="valid")
 
    k = min(5, len(results.score_log))
 
    ax.scatter(results.frame_log, results.score_log,
               s=12, alpha=0.35, color="gold", zorder=2,
               label="Mean score (100 ep window)")
    ax.plot(results.frame_log[k - 1:], _smooth(results.score_log, k),
            color="gold", linewidth=2.5, zorder=3,
            label=f"Smoothed ({k}-point moving avg)")
    
    if results.solved_frame is not None:
        ax.axvline(results.solved_frame, color="limegreen", linestyle=":",
                   linewidth=1.5, zorder=1,
                   label=f"Solved at frame {results.solved_frame:,}")
 
    ax.set_xlabel("Environment frames", fontsize=12)
    ax.set_ylabel("Mean episode score (last 100 ep)", fontsize=12)
    ax.set_title(f"Rainbow (no Dueling) on {cfg.env_id} — Score vs Frames",
                 fontsize=13)
    ax.set_xlim(0, cfg.max_frames)
    ax.set_ylim(0, 520)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()



