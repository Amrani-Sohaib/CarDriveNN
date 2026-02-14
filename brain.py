"""
LSTM Actor-Critic with PPO (Proximal Policy Optimization).
==========================================================
- Actor:  LSTM-based policy with SquashedNormal (tanh-squashed Gaussian)
- Critic: LSTM-based value network (shared backbone)
- Memory: LSTM hidden state carries temporal context across timesteps
- Training: PPO with GAE, sequential LSTM mini-batches (chunked)
- Obs:    Running observation normaliser (Welford online algorithm)

Inputs  (15):  12 radars + speed + angle_to_cp + curvature
Outputs  (2):  steering [-1, +1],  throttle [-1, +1]
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
#  Running Observation Normaliser (Welford)
# =====================================================================
class RunningNormalizer:
    """Online running-mean / running-std normaliser (Welford algorithm)."""

    def __init__(self, shape, clip=5.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x
        batch_var = np.zeros_like(self.mean)
        batch_count = 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        new_var = m2 / tot
        self.mean = new_mean
        self.var = new_var
        self.count = tot

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        return np.clip(
            (x - self.mean.astype(np.float32))
            / (np.sqrt(self.var).astype(np.float32) + 1e-8),
            -self.clip,
            self.clip,
        )

    def state_dict(self):
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
        }

    def load_state_dict(self, d):
        self.mean = np.array(d["mean"], dtype=np.float64)
        self.var = np.array(d["var"], dtype=np.float64)
        self.count = float(d["count"])


# =====================================================================
#  Squashed Normal Distribution  (tanh-squash)
# =====================================================================
class SquashedNormal:
    """
    Normal distribution followed by tanh squashing.
    Correct log_prob with change-of-variable Jacobian correction:
        log pi(a) = log N(u; mu, sigma) - sum log(1 - tanh(u)^2)
    where a = tanh(u).
    """

    def __init__(self, mean, std):
        self.normal = torch.distributions.Normal(mean, std)

    def sample(self):
        u = self.normal.rsample()  # reparameterised sample
        a = torch.tanh(u)
        return a, u  # return both for log_prob

    def log_prob(self, u):
        """log_prob given the *pre-tanh* sample u."""
        log_p = self.normal.log_prob(u)
        # Jacobian correction: log(1 - tanh(u)^2) with numerical stability
        log_p = log_p - torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6)
        return log_p.sum(dim=-1)

    def entropy_approx(self):
        """Approximate entropy (Gaussian entropy as proxy)."""
        return self.normal.entropy().sum(dim=-1)


# =====================================================================
#  Actor-Critic LSTM Network
# =====================================================================
class ActorCriticLSTM(nn.Module):
    """
    Shared LSTM backbone -> Actor head (SquashedNormal) + Critic head.
    In(15) -> FC(128) -> FC(128) -> LSTM(128) -> Actor(2) + Critic(1)
    """

    INPUT_DIM = 15
    HIDDEN_FC = 128
    LSTM_SIZE = 128
    LSTM_LAYERS = 1
    ACTION_DIM = 2  # steering, throttle

    def __init__(self):
        super().__init__()

        # shared feature extractor
        self.fc1 = nn.Linear(self.INPUT_DIM, self.HIDDEN_FC)
        self.fc2 = nn.Linear(self.HIDDEN_FC, self.HIDDEN_FC)

        # LSTM for temporal memory
        self.lstm = nn.LSTM(
            input_size=self.HIDDEN_FC,
            hidden_size=self.LSTM_SIZE,
            num_layers=self.LSTM_LAYERS,
            batch_first=True,
        )

        # actor head: outputs pre-tanh mean
        self.actor_fc = nn.Linear(self.LSTM_SIZE, 64)
        self.actor_out = nn.Linear(64, self.ACTION_DIM)

        # learnable log_std (separate param, not squashed)
        self.actor_log_std = nn.Parameter(torch.full((self.ACTION_DIM,), 0.0))

        # critic head: state value
        self.critic_fc = nn.Linear(self.LSTM_SIZE, 64)
        self.critic_out = nn.Linear(64, 1)

        self._init_weights()
        self.to(DEVICE)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # small init for actor output -> conservative initial policy
        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)
        nn.init.constant_(self.actor_out.bias, 0)

    def init_hidden(self, batch_size=1):
        """Create fresh LSTM hidden state."""
        h = torch.zeros(
            self.LSTM_LAYERS, batch_size, self.LSTM_SIZE, device=DEVICE
        )
        c = torch.zeros(
            self.LSTM_LAYERS, batch_size, self.LSTM_SIZE, device=DEVICE
        )
        return (h, c)

    def forward(self, x, hidden):
        """
        x:      (batch, seq_len, INPUT_DIM) or (INPUT_DIM,)
        hidden: (h, c) LSTM state
        Returns: pre_tanh_mean, std, value, new_hidden
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # shared trunk
        z = torch.relu(self.fc1(x))
        z = torch.relu(self.fc2(z))

        lstm_out, new_hidden = self.lstm(z, hidden)

        # if sequence length > 1: keep all timesteps for sequential PPO
        if lstm_out.shape[1] > 1:
            feat = lstm_out  # (B, T, LSTM_SIZE)
        else:
            feat = lstm_out[:, -1:, :]  # (B, 1, LSTM_SIZE)

        # actor -- pre-tanh mean (unbounded)
        a = torch.relu(self.actor_fc(feat))
        pre_tanh_mean = self.actor_out(a)
        std = torch.exp(
            torch.clamp(self.actor_log_std, -3.0, 0.5)
        ).expand_as(pre_tanh_mean)

        # critic
        v = torch.relu(self.critic_fc(feat))
        value = self.critic_out(v)

        return pre_tanh_mean, std, value, new_hidden

    # -- inference helpers ------------------------------------------------
    @torch.no_grad()
    def act(self, obs_np, hidden):
        """Select action via SquashedNormal.
        Returns (action_np, log_prob, value, new_hidden, pre_tanh_np)."""
        obs_t = torch.FloatTensor(obs_np).to(DEVICE)
        mean, std, value, new_hidden = self.forward(obs_t, hidden)

        dist = SquashedNormal(mean, std)
        action, u = dist.sample()  # action = tanh(u)
        log_prob = dist.log_prob(u)  # corrected log_prob

        return (
            action.squeeze(0).squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.squeeze().item(),
            new_hidden,
            u.squeeze(0).squeeze(0).cpu().numpy(),
        )

    @torch.no_grad()
    def act_deterministic(self, obs_np, hidden):
        """Deterministic action for test mode: tanh(mean)."""
        obs_t = torch.FloatTensor(obs_np).to(DEVICE)
        mean, _, value, new_hidden = self.forward(obs_t, hidden)
        action = torch.tanh(mean)
        return (
            action.squeeze(0).squeeze(0).cpu().numpy(),
            value.squeeze().item(),
            new_hidden,
        )

    @torch.no_grad()
    def get_value(self, obs_np, hidden):
        obs_t = torch.FloatTensor(obs_np).to(DEVICE)
        _, _, value, _ = self.forward(obs_t, hidden)
        return value.squeeze().item()

    def evaluate_actions_sequential(self, obs_seq, pre_tanh_actions, hidden):
        """
        Evaluate a *sequence* for PPO update.
        obs_seq:          (B, T, INPUT_DIM)
        pre_tanh_actions: (B, T, ACTION_DIM) -- pre-tanh u values from rollout
        hidden:           (h, c) initial hidden for each sequence chunk
        Returns: log_probs (B, T), values (B, T), entropy (B, T)
        """
        mean, std, values, _ = self.forward(obs_seq, hidden)

        dist = SquashedNormal(mean, std)
        log_probs = dist.log_prob(pre_tanh_actions)
        entropy = dist.entropy_approx()

        return log_probs, values.squeeze(-1), entropy

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_architecture_str(self):
        return (
            f"In({self.INPUT_DIM}) > FC({self.HIDDEN_FC})x2 "
            f"> LSTM({self.LSTM_SIZE}) > Actor({self.ACTION_DIM}) + Critic(1)"
        )

    # -- visualisation helpers -------------------------------------------
    def get_layer_sizes(self):
        return [
            self.INPUT_DIM,
            self.HIDDEN_FC,
            self.HIDDEN_FC,
            self.LSTM_SIZE,
            64,
            self.ACTION_DIM,
        ]

    @torch.no_grad()
    def get_activations(self, obs, hidden):
        """Per-layer activations for GUI visualisation."""
        obs_np = np.array(obs, dtype=np.float32)
        obs_t = (
            torch.FloatTensor(obs_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
        )

        activations = [obs_np.tolist()]

        z1 = torch.relu(self.fc1(obs_t))
        activations.append(z1.squeeze().cpu().numpy().tolist())

        z2 = torch.relu(self.fc2(z1))
        activations.append(z2.squeeze().cpu().numpy().tolist())

        lstm_out, _ = self.lstm(z2, hidden)
        last = lstm_out[:, -1, :]
        activations.append(last.squeeze().cpu().numpy().tolist())

        a = torch.relu(self.actor_fc(last))
        activations.append(a.squeeze().cpu().numpy().tolist())

        out = torch.tanh(self.actor_out(a))
        activations.append(out.squeeze().cpu().numpy().tolist())

        return activations

    @torch.no_grad()
    def get_weight_stats(self):
        """Per-layer weight statistics for visualisation."""
        layers = [self.fc1, self.fc2, self.lstm, self.actor_fc, self.actor_out]
        names = ["FC1", "FC2", "LSTM", "ActorFC", "ActorOut"]
        stats = []
        for name, layer in zip(names, layers):
            params = list(layer.parameters())
            all_w = torch.cat([p.detach().flatten() for p in params])
            stats.append(
                {
                    "name": name,
                    "mean_w": float(torch.mean(torch.abs(all_w))),
                    "std_w": float(torch.std(all_w)),
                    "max_w": float(torch.max(torch.abs(all_w))),
                }
            )
        return stats


# =====================================================================
#  Trajectory Buffer  (sequential chunks for LSTM)
# =====================================================================
SEQ_LEN = 32  # LSTM chunk length for sequential mini-batches


class TrajectoryBuffer:
    """
    Stores transitions **with hidden states**.
    For PPO update: yields sequential chunks of length SEQ_LEN
    so the LSTM is trained on real temporal sequences.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.pre_tanh_actions = []  # u values (pre-tanh) for correct log_prob
        self.actions = []  # tanh(u) -- actual actions applied
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.hiddens_h = []  # LSTM h at *start* of each step
        self.hiddens_c = []  # LSTM c at *start* of each step

    def push(self, obs, action, pre_tanh, log_prob, reward, value, done, hidden):
        """Store one transition including the LSTM hidden state."""
        self.observations.append(np.array(obs, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.pre_tanh_actions.append(np.array(pre_tanh, dtype=np.float32))
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(float(done))
        # hidden: (h, c) each shape (num_layers, 1, lstm_size)
        self.hiddens_h.append(hidden[0].squeeze(1).cpu().numpy())
        self.hiddens_c.append(hidden[1].squeeze(1).cpu().numpy())

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        """Generalised Advantage Estimation (handles truncation via dones)."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = last_value
            else:
                next_val = self.values[t + 1]
            next_non_done = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * next_val * next_non_done
                - self.values[t]
            )
            gae = delta + gamma * lam * next_non_done * gae
            advantages[t] = gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return returns, advantages

    def make_sequential_batches(self, returns, advantages, mini_batch_seqs=8):
        """
        Split rollout into chunks of SEQ_LEN and yield mini-batches
        of ``mini_batch_seqs`` sequences for sequential LSTM training.
        """
        n = len(self.rewards)
        if n < SEQ_LEN:
            return

        # build non-overlapping chunks
        chunks = []
        start = 0
        while start + SEQ_LEN <= n:
            chunks.append(start)
            start += SEQ_LEN

        if len(chunks) == 0:
            return

        np.random.shuffle(chunks)

        obs_all = np.array(self.observations, dtype=np.float32)
        pre_tanh_all = np.array(self.pre_tanh_actions, dtype=np.float32)
        old_log_all = np.array(self.log_probs, dtype=np.float32)
        ret_all = returns
        adv_all = advantages
        done_all = np.array(self.dones, dtype=np.float32)
        hid_h_all = np.array(self.hiddens_h)
        hid_c_all = np.array(self.hiddens_c)

        for mb_start in range(0, len(chunks), mini_batch_seqs):
            mb_chunks = chunks[mb_start : mb_start + mini_batch_seqs]
            B = len(mb_chunks)
            T = SEQ_LEN

            mb_obs = np.zeros(
                (B, T, ActorCriticLSTM.INPUT_DIM), dtype=np.float32
            )
            mb_pre_tanh = np.zeros(
                (B, T, ActorCriticLSTM.ACTION_DIM), dtype=np.float32
            )
            mb_old_log = np.zeros((B, T), dtype=np.float32)
            mb_ret = np.zeros((B, T), dtype=np.float32)
            mb_adv = np.zeros((B, T), dtype=np.float32)
            mb_mask = np.ones((B, T), dtype=np.float32)

            mb_h = np.zeros(
                (ActorCriticLSTM.LSTM_LAYERS, B, ActorCriticLSTM.LSTM_SIZE),
                dtype=np.float32,
            )
            mb_c = np.zeros(
                (ActorCriticLSTM.LSTM_LAYERS, B, ActorCriticLSTM.LSTM_SIZE),
                dtype=np.float32,
            )

            for bi, c_start in enumerate(mb_chunks):
                c_end = c_start + T
                mb_obs[bi] = obs_all[c_start:c_end]
                mb_pre_tanh[bi] = pre_tanh_all[c_start:c_end]
                mb_old_log[bi] = old_log_all[c_start:c_end]
                mb_ret[bi] = ret_all[c_start:c_end]
                mb_adv[bi] = adv_all[c_start:c_end]

                # use hidden from the start of this chunk
                mb_h[:, bi, :] = hid_h_all[c_start]
                mb_c[:, bi, :] = hid_c_all[c_start]

                # mask: zero out steps after a done inside the chunk
                for ti in range(T):
                    if done_all[c_start + ti] > 0.5 and ti < T - 1:
                        mb_mask[bi, ti + 1 :] = 0.0
                        break

            yield {
                "obs": torch.FloatTensor(mb_obs).to(DEVICE),
                "pre_tanh": torch.FloatTensor(mb_pre_tanh).to(DEVICE),
                "old_log": torch.FloatTensor(mb_old_log).to(DEVICE),
                "returns": torch.FloatTensor(mb_ret).to(DEVICE),
                "advantages": torch.FloatTensor(mb_adv).to(DEVICE),
                "mask": torch.FloatTensor(mb_mask).to(DEVICE),
                "hidden": (
                    torch.FloatTensor(mb_h).to(DEVICE),
                    torch.FloatTensor(mb_c).to(DEVICE),
                ),
            }


# =====================================================================
#  PPO Trainer
# =====================================================================
class PPOTrainer:
    """Proximal Policy Optimization with sequential LSTM training."""

    def __init__(self, network: ActorCriticLSTM):
        self.network = network
        self.optimizer = optim.Adam(
            network.parameters(), lr=3e-4, eps=1e-5
        )

        # PPO hyperparameters
        self.clip_eps = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.entropy_coef = 0.05
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        self.mini_batch_seqs = 8  # number of sequence chunks per mini-batch

        # stats
        self.episode = 0
        self.global_step = 0
        self.total_updates = 0
        self.best_fitness = -1e9
        self.improvements = 0
        self.last_pg_loss = 0.0
        self.last_v_loss = 0.0
        self.last_entropy = 0.0
        self.recent_rewards = []

        # observation normaliser
        self.obs_normalizer = RunningNormalizer(
            shape=(ActorCriticLSTM.INPUT_DIM,)
        )

        # buffer
        self.buffer = TrajectoryBuffer()

    def update(self, last_value):
        """Run PPO update with sequential LSTM mini-batches."""
        if len(self.buffer) < SEQ_LEN:
            self.buffer.clear()
            return

        returns, advantages = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )

        # normalise advantages globally
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        total_pg = total_v = total_ent = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.make_sequential_batches(
                returns, advantages, self.mini_batch_seqs
            ):
                mb_obs = batch["obs"]  # (B, T, input_dim)
                mb_pre_tanh = batch["pre_tanh"]  # (B, T, action_dim)
                mb_old_log = batch["old_log"]  # (B, T)
                mb_ret = batch["returns"]  # (B, T)
                mb_adv = batch["advantages"]  # (B, T)
                mb_mask = batch["mask"]  # (B, T)
                hidden = batch["hidden"]  # detached (h, c)

                # forward through entire sequence
                new_log, values, entropy = (
                    self.network.evaluate_actions_sequential(
                        mb_obs,
                        mb_pre_tanh,
                        (hidden[0].detach(), hidden[1].detach()),
                    )
                )

                # masked PPO loss
                ratio = torch.exp(new_log - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = (
                    torch.clamp(
                        ratio, 1 - self.clip_eps, 1 + self.clip_eps
                    )
                    * mb_adv
                )
                pg_loss = (
                    -(torch.min(surr1, surr2) * mb_mask).sum()
                    / mb_mask.sum()
                )

                v_loss = (
                    0.5
                    * ((mb_ret - values).pow(2) * mb_mask).sum()
                    / mb_mask.sum()
                )
                ent_loss = -(entropy * mb_mask).sum() / mb_mask.sum()

                loss = (
                    pg_loss
                    + self.value_coef * v_loss
                    + self.entropy_coef * ent_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_pg += pg_loss.item()
                total_v += v_loss.item()
                total_ent += entropy.mean().item()
                num_updates += 1

        self.total_updates += num_updates
        self.last_pg_loss = total_pg / max(1, num_updates)
        self.last_v_loss = total_v / max(1, num_updates)
        self.last_entropy = total_ent / max(1, num_updates)
        self.buffer.clear()

    def get_stats(self) -> dict:
        avg_r = (
            float(np.mean(self.recent_rewards[-30:]))
            if self.recent_rewards
            else 0
        )
        return {
            "episode": self.episode,
            "best_fitness": self.best_fitness,
            "current_fitness": 0,
            "improvements": self.improvements,
            "params_count": self.network.count_params(),
            "total_updates": self.total_updates,
            "avg_reward": avg_r,
            "pg_loss": self.last_pg_loss,
            "v_loss": self.last_v_loss,
            "entropy": self.last_entropy,
            "lr": self.optimizer.param_groups[0]["lr"],
            "global_step": self.global_step,
            "device": str(DEVICE),
        }

    # -- Save / Load  (torch checkpoint) ----------------------------------
    def save(self, filepath: str):
        """Save a complete training checkpoint with torch.save."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": self.episode,
            "global_step": self.global_step,
            "total_updates": self.total_updates,
            "best_fitness": float(self.best_fitness),
            "improvements": self.improvements,
            "recent_rewards": self.recent_rewards[-100:],
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "input_dim": ActorCriticLSTM.INPUT_DIM,
            "action_dim": ActorCriticLSTM.ACTION_DIM,
            "lstm_size": ActorCriticLSTM.LSTM_SIZE,
        }
        torch.save(checkpoint, filepath)
        return filepath

    def load_checkpoint(self, filepath: str):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(
            filepath, map_location=DEVICE, weights_only=False
        )
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode = checkpoint.get("episode", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.total_updates = checkpoint.get("total_updates", 0)
        self.best_fitness = checkpoint.get("best_fitness", -1e9)
        self.improvements = checkpoint.get("improvements", 0)
        self.recent_rewards = checkpoint.get("recent_rewards", [])
        if "obs_normalizer" in checkpoint:
            self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
        self.network.to(DEVICE)
        print(
            f"  Checkpoint loaded: ep={self.episode}, "
            f"step={self.global_step}, best={self.best_fitness:.0f}"
        )
        return checkpoint

    @staticmethod
    def load_for_test(filepath: str):
        """Load a checkpoint for inference / test only."""
        checkpoint = torch.load(
            filepath, map_location=DEVICE, weights_only=False
        )
        network = ActorCriticLSTM()
        network.load_state_dict(checkpoint["model_state_dict"])
        network.eval()
        network.to(DEVICE)

        obs_normalizer = RunningNormalizer(
            shape=(ActorCriticLSTM.INPUT_DIM,)
        )
        if "obs_normalizer" in checkpoint:
            obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])

        info = {
            "episode": checkpoint.get("episode", 0),
            "global_step": checkpoint.get("global_step", 0),
            "best_fitness": checkpoint.get("best_fitness", 0),
            "total_updates": checkpoint.get("total_updates", 0),
            "improvements": checkpoint.get("improvements", 0),
        }
        return network, info, obs_normalizer
