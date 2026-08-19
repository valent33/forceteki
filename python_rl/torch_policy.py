from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# Global action-slot count; matches SWUEnv.action_space (Discrete(max_action_space)).
DEFAULT_MAX_ACTIONS = 100
# Logit forced onto illegal action slots before the softmax.
ILLEGAL_LOGIT = -1e9


class DualHeadNetwork(nn.Module):
    """
    Shared-trunk actor-critic for the SWU State Tensor.

    obs (OBS_DIM floats)
      → MLP trunk (Linear → GELU → LayerNorm → Dropout per hidden layer)
        ├─ policy head: Linear → MAX_ACTIONS raw logits
        └─ value head:  Linear → GELU → Linear → Tanh() ∈ [-1, 1]
    """

    def __init__(
        self,
        obs_size: int = 2386,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        hidden_sizes: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_actions = max_actions

        trunk_layers: list[nn.Module] = []
        input_size = obs_size
        for hidden_size in hidden_sizes:
            trunk_layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
            ])
            input_size = hidden_size

        # Named `obs_encoder` so existing checkpoint-peeking code
        # (`state_dict["obs_encoder.0.weight"].shape[1]`) keeps working.
        self.obs_encoder = nn.Sequential(*trunk_layers)

        self.policy_head = nn.Linear(input_size, max_actions)
        self.value_head = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, obs_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits [B, MAX_ACTIONS], state_value [B, 1] ∈ [-1, 1])."""
        features = self.obs_encoder(obs_batch)
        action_logits = self.policy_head(features)
        state_value = self.value_head(features)
        return action_logits, state_value


class TorchPolicy:
    """
    Actor-critic policy for SWU.

    - π head: raw logits over MAX_ACTIONS global action slots. The dynamic
      `legal_action_mask` from the environment sets illegal slots to -1e9
      before the softmax (dynamic action masking).
    - V head: tanh-bounded scalar in [-1, 1] (+1 = win, -1 = loss) estimating
      the expected game outcome from the current state.
    """

    def __init__(
        self,
        obs_size: int = 2386,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        hidden_sizes: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        lr: float = 1e-3,
        device: str = "cpu",
        # Backward-compatible no-ops from the old candidate-scoring policy:
        action_feature_size: int = 40,
        action_size: int | None = None,
    ):
        self.device = torch.device(device)
        if action_size is not None:
            max_actions = int(action_size)  # older trainer code passed a fixed action size
        self.obs_size = int(obs_size)
        self.max_actions = int(max_actions)
        # Kept for legacy attribute access (e.g. SnapshotOpponent).
        self.action_feature_size = int(action_feature_size)
        self.net = DualHeadNetwork(
            obs_size=self.obs_size,
            max_actions=self.max_actions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    # ── Core forward passes ──────────────────────────────────────────────────
    def forward(self, obs_batch: torch.Tensor, legal_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_logits, state_value); illegal slots → -1e9 when a mask is given."""
        logits, value = self.net(obs_batch.to(self.device).float())
        if legal_mask is not None:
            logits = self._apply_mask(logits, legal_mask)
        return logits, value

    def masked_logits(self, obs_tensor: torch.Tensor, available_actions=None, legal_mask=None):
        """
        π-head logits with every unavailable action slot forced to -1e9.

        Returns (logits [MAX_ACTIONS], state_value []).
        """
        logits, value = self._compute_logits(obs_tensor)
        n = len(available_actions) if available_actions is not None else self.max_actions
        mask_t = self._build_mask_tensor(n, legal_mask)
        logits = torch.where(mask_t > 0.0, logits, torch.full_like(logits, ILLEGAL_LOGIT))
        return logits, value

    # ── Acting ───────────────────────────────────────────────────────────────
    def select_action(self, obs_tensor: torch.Tensor, available_actions=None, legal_mask=None):
        """
        Sample an action from the masked π head.

        Returns (action_index, log_prob, state_value). `action_index` indexes
        into `available_actions`, so it can be passed directly to env.step().
        Returns (None, None, None) when no actions are available.
        """
        if available_actions is not None and len(available_actions) == 0:
            return None, None, None

        logits, value = self.masked_logits(obs_tensor, available_actions, legal_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action_tensor = dist.sample()
        return int(action_tensor.item()), dist.log_prob(action_tensor), value.detach()

    def evaluate(self, obs_batch: torch.Tensor, action_indices: torch.Tensor):
        """(log_probs, values) for a batch of stored transitions (training helper)."""
        obs = obs_batch.to(self.device).float()
        logits, values = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        indices = action_indices.to(self.device).long()
        return dist.log_prob(indices), values

    # ── Training ─────────────────────────────────────────────────────────────
    def update(self, logps, returns, values=None, value_coef: float = 0.5, entropy_coef: float = 0.01):
        """
        Advantage Actor-Critic update.

        Loss = L_policy + value_coef · L_value − entropy_coef · H_entropy
          L_policy  = mean(−log π(a) · (return − V(s)))      (advantage baseline)
          L_value   = MSE(V(s), return)                      (critic)
          H_entropy = mean(−log π(a))                        (single-sample estimator)

        Returns (total_loss, policy_loss, value_loss, entropy) as floats.
        """
        if not logps or len(returns) == 0:
            return 0.0, 0.0, 0.0, 0.0

        rets = torch.stack([self._as_tensor(r).to(self.device) for r in returns])
        lp_stack = torch.stack([lp.to(self.device) for lp in logps])
        n = max(1, len(logps))

        if values is not None and len(values) == len(logps):
            vals = torch.stack([self._as_tensor(v).to(self.device) for v in values])
            advantages = (rets - vals).detach()
            value_loss = F.mse_loss(vals, rets)
        else:
            advantages = rets
            value_loss = torch.tensor(0.0, device=self.device)

        policy_loss = torch.sum(-lp_stack * advantages) / n
        entropy = torch.mean(-lp_stack)  # E[-log π(a)] = H(π) under the policy

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return (
            float(loss.item()),
            float(policy_loss.item()),
            float(value_loss.item()),
            float(entropy.item()),
        )

    # ── Internals ────────────────────────────────────────────────────────────
    def _compute_logits(self, obs_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = obs_tensor.to(self.device).float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        logits, value = self.net(obs)
        return logits.squeeze(0), value.squeeze(-1).squeeze(0)

    def _build_mask_tensor(self, n: int, legal_mask) -> torch.Tensor:
        mask = torch.zeros(self.max_actions, dtype=torch.float32, device=self.device)
        if legal_mask is not None and len(legal_mask) >= n:
            for i in range(n):
                if bool(legal_mask[i]):
                    mask[i] = 1.0
        else:
            mask[:n] = 1.0
        if mask.sum().item() <= 0.0:  # safety: never mask everything
            mask[:n] = 1.0
        return mask

    def _apply_mask(self, logits: torch.Tensor, legal_mask) -> torch.Tensor:
        mask_t = torch.tensor(
            [1.0 if i < len(legal_mask) and bool(legal_mask[i]) else 0.0 for i in range(logits.shape[-1])],
            dtype=torch.float32,
            device=logits.device,
        )
        if mask_t.sum().item() <= 0.0:
            mask_t = torch.ones_like(mask_t)
        return torch.where(mask_t > 0.0, logits, torch.full_like(logits, ILLEGAL_LOGIT))

    @staticmethod
    def _as_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.reshape(()).float()
        return torch.tensor(float(value), dtype=torch.float32)
