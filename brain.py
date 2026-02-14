"""
LSTM Actor-Critic with PPO (Proximal Policy Optimization).
==========================================================
- Actor:  LSTM-based policy network with continuous outputs
          (steering: tanh, throttle: tanh)
- Critic: LSTM-based value network (shared backbone)
- Memory: LSTM hidden state carries temporal context across timesteps
- Training: PPO with GAE (Generalized Advantage Estimation)

Inputs (15):
  12 radars + speed + angle_to_cp + curvature
Outputs (2):
  steering [-1, +1],  throttle [-1, +1]
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json, os

DEVICE = torch.device("cpu")


# =====================================================================
#  Actor-Critic LSTM Network
# =====================================================================
class ActorCriticLSTM(nn.Module):
    """
    Shared LSTM backbone -> Actor head (continuous) + Critic head.
    In(15) -> FC(128) -> FC(128) -> LSTM(128) -> Actor(2) + Critic(1)
    """

    INPUT_DIM   = 15
    HIDDEN_FC   = 128
    LSTM_SIZE   = 128
    LSTM_LAYERS = 1
    ACTION_DIM  = 2   # steering, throttle

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

        # actor head: mean of actions
        self.actor_fc = nn.Linear(self.LSTM_SIZE, 64)
        self.actor_out = nn.Linear(64, self.ACTION_DIM)

        # actor log_std (learnable)
        self.actor_log_std = nn.Parameter(torch.zeros(self.ACTION_DIM))

        # critic head: state value
        self.critic_fc = nn.Linear(self.LSTM_SIZE, 64)
        self.critic_out = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # small init for actor output -> more exploration early
        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)
        nn.init.constant_(self.actor_out.bias, 0)

    def init_hidden(self, batch_size=1):
        """Create fresh LSTM hidden state."""
        h = torch.zeros(self.LSTM_LAYERS, batch_size, self.LSTM_SIZE, device=DEVICE)
        c = torch.zeros(self.LSTM_LAYERS, batch_size, self.LSTM_SIZE, device=DEVICE)
        return (h, c)

    def forward(self, x, hidden):
        """
        x:      (batch, seq_len, INPUT_DIM) or (INPUT_DIM,)
        hidden: (h, c) LSTM state
        Returns: action_mean, action_std, value, new_hidden
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # shared trunk
        z = torch.relu(self.fc1(x))
        z = torch.relu(self.fc2(z))

        lstm_out, new_hidden = self.lstm(z, hidden)
        last = lstm_out[:, -1, :]          # (batch, LSTM_SIZE)

        # actor
        a = torch.relu(self.actor_fc(last))
        action_mean = torch.tanh(self.actor_out(a))
        action_std  = torch.exp(torch.clamp(self.actor_log_std, -5, 2)).expand_as(action_mean)

        # critic
        v = torch.relu(self.critic_fc(last))
        value = self.critic_out(v)

        return action_mean, action_std, value, new_hidden

    # -- inference helpers ------------------------------------------------
    @torch.no_grad()
    def act(self, obs, hidden):
        """Select action for a single observation.
        Returns (action_np, log_prob, value, new_hidden)."""
        obs_t = torch.FloatTensor(obs).to(DEVICE)
        mean, std, value, new_hidden = self.forward(obs_t, hidden)

        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
            new_hidden,
        )

    @torch.no_grad()
    def act_deterministic(self, obs, hidden):
        """Deterministic action (for test mode)."""
        obs_t = torch.FloatTensor(obs).to(DEVICE)
        mean, _, value, new_hidden = self.forward(obs_t, hidden)
        action = torch.clamp(mean, -1.0, 1.0)
        return action.squeeze(0).cpu().numpy(), value.item(), new_hidden

    @torch.no_grad()
    def get_value(self, obs, hidden):
        obs_t = torch.FloatTensor(obs).to(DEVICE)
        _, _, value, _ = self.forward(obs_t, hidden)
        return value.item()

    def evaluate_actions(self, obs_batch, actions_batch, hidden):
        """Evaluate a batch for PPO update."""
        mean, std, values, _ = self.forward(obs_batch, hidden)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions_batch).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, values.squeeze(-1), entropy

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_architecture_str(self):
        return (
            f"In({self.INPUT_DIM}) > FC({self.HIDDEN_FC})x2 "
            f"> LSTM({self.LSTM_SIZE}) > Actor({self.ACTION_DIM}) + Critic(1)"
        )

    # -- visualization helpers -------------------------------------------
    def get_layer_sizes(self):
        return [self.INPUT_DIM, self.HIDDEN_FC, self.HIDDEN_FC,
                self.LSTM_SIZE, 64, self.ACTION_DIM]

    @torch.no_grad()
    def get_activations(self, obs, hidden):
        """Return per-layer activations for GUI visualization."""
        obs_np = np.array(obs, dtype=np.float32)
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

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
        """Per-layer weight statistics for visualization."""
        layers = [self.fc1, self.fc2, self.lstm, self.actor_fc, self.actor_out]
        names  = ["FC1", "FC2", "LSTM", "ActorFC", "ActorOut"]
        stats = []
        for name, layer in zip(names, layers):
            params = list(layer.parameters())
            all_w = torch.cat([p.detach().flatten() for p in params])
            stats.append({
                "name": name,
                "mean_w": float(torch.mean(torch.abs(all_w))),
                "std_w": float(torch.std(all_w)),
                "max_w": float(torch.max(torch.abs(all_w))),
            })
        return stats


# =====================================================================
#  Trajectory Buffer (PPO rollout storage)
# =====================================================================
class TrajectoryBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions      = []
        self.log_probs    = []
        self.rewards      = []
        self.values       = []
        self.dones        = []

    def push(self, obs, action, log_prob, reward, value, done):
        self.observations.append(np.array(obs, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(float(done))

    def __len__(self):
        return len(self.rewards)

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        """Generalized Advantage Estimation."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = last_value
            else:
                next_val = self.values[t + 1]
            next_non_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * next_non_done - self.values[t]
            gae = delta + gamma * lam * next_non_done * gae
            advantages[t] = gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return returns, advantages


# =====================================================================
#  PPO Trainer
# =====================================================================
class PPOTrainer:
    """Proximal Policy Optimization trainer."""

    def __init__(self, network: ActorCriticLSTM):
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=3e-4, eps=1e-5)

        # PPO hyperparameters
        self.clip_eps       = 0.2
        self.gamma          = 0.99
        self.gae_lambda     = 0.95
        self.entropy_coef   = 0.01
        self.value_coef     = 0.5
        self.max_grad_norm  = 0.5
        self.ppo_epochs     = 4
        self.mini_batch_size = 64

        # stats
        self.episode        = 0
        self.total_updates  = 0
        self.best_fitness   = -1e9
        self.improvements   = 0
        self.last_pg_loss   = 0.0
        self.last_v_loss    = 0.0
        self.last_entropy   = 0.0
        self.recent_rewards = []

        # buffer
        self.buffer = TrajectoryBuffer()

    def update(self, last_value):
        """Run PPO update on the collected buffer."""
        if len(self.buffer) < self.mini_batch_size:
            self.buffer.clear()
            return

        returns, advantages = self.buffer.compute_gae(
            last_value, self.gamma, self.gae_lambda
        )

        obs_t     = torch.FloatTensor(np.array(self.buffer.observations)).to(DEVICE)
        act_t     = torch.FloatTensor(np.array(self.buffer.actions)).to(DEVICE)
        old_log_t = torch.FloatTensor(np.array(self.buffer.log_probs)).to(DEVICE)
        ret_t     = torch.FloatTensor(returns).to(DEVICE)
        adv_t     = torch.FloatTensor(advantages).to(DEVICE)
        adv_t     = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(self.buffer)
        total_pg = total_v = total_ent = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                idx = perm[start:end]

                mb_obs = obs_t[idx]
                mb_act = act_t[idx]
                mb_old_log = old_log_t[idx]
                mb_ret = ret_t[idx]
                mb_adv = adv_t[idx]

                hidden = self.network.init_hidden(len(idx))
                new_log, values, entropy = self.network.evaluate_actions(
                    mb_obs, mb_act, hidden
                )

                ratio = torch.exp(new_log - mb_old_log)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                v_loss = 0.5 * (mb_ret - values).pow(2).mean()
                ent_loss = -entropy.mean()

                loss = pg_loss + self.value_coef * v_loss + self.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg  += pg_loss.item()
                total_v   += v_loss.item()
                total_ent += entropy.mean().item()
                num_updates += 1

        self.total_updates += num_updates
        self.last_pg_loss  = total_pg / max(1, num_updates)
        self.last_v_loss   = total_v / max(1, num_updates)
        self.last_entropy  = total_ent / max(1, num_updates)
        self.buffer.clear()

    def get_stats(self) -> dict:
        avg_r = float(np.mean(self.recent_rewards[-30:])) if self.recent_rewards else 0
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
        }

    # -- Save / Load -------------------------------------------------------
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "model_state": {k: v.tolist() for k, v in self.network.state_dict().items()},
            "best_fitness": float(self.best_fitness),
            "episode": self.episode,
            "improvements": self.improvements,
            "total_updates": self.total_updates,
            "input_dim": ActorCriticLSTM.INPUT_DIM,
            "action_dim": ActorCriticLSTM.ACTION_DIM,
            "lstm_size": ActorCriticLSTM.LSTM_SIZE,
        }
        with open(filepath, "w") as f:
            json.dump(data, f)
        return filepath

    @staticmethod
    def load(filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        network = ActorCriticLSTM()
        state_dict = {k: torch.tensor(v) for k, v in data["model_state"].items()}
        network.load_state_dict(state_dict)
        network.eval()
        info = {
            "best_fitness": data.get("best_fitness", 0),
            "episode": data.get("episode", 0),
            "improvements": data.get("improvements", 0),
            "total_updates": data.get("total_updates", 0),
        }
        return network, info
