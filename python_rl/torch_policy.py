from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# Global action-slot count; matches SWUEnv.action_space (Discrete(max_action_space)).
DEFAULT_MAX_ACTIONS = 100
# Logit forced onto illegal action slots before the softmax.
ILLEGAL_LOGIT = -1e9

# ── Per-action feature vector shared by the env, the trainer and the GUI bot.
# The policy scores each candidate action as Q(state_latent, action_features)
# instead of by its position in the 100-slot list.
ACTION_FEATURE_DIM = 64
_ACTION_TYPE_INDEX = {
    "clickCard": 0,
    "clickPrompt": 1,
    "perCardMenuButton": 2,
    "displayCardClick": 3,
    "macro_resource_cards": 4,
    "macro_select_all_cards": 5,
    "statefulPromptResults": 6,
    "menuButton": 7,
}


def build_action_features(action: dict, cost: float = 0.0) -> list[float]:
    """Build the 64-dim feature vector describing ONE candidate action.

    Layout:
      0-7   one-hot action type
      8-13  is_pass / is_claim / is_done / is_cancel / is_attack / is_card
      14-19 is_leader / is_base / is_unit / is_exhausted / is_friendly / is_playable
      20-22 is_stateful / is_macro / is_dropdown
      23-25 card_power / card_hp / cost
      26-41 uuid hash one-hot (16 buckets)
      42-43 is_self_target / is_opponent_target (button targeting semantics)
      44-63 reserved
    """
    vec = [0.0] * ACTION_FEATURE_DIM
    action_type = str(action.get("actionType") or "")
    if action_type in _ACTION_TYPE_INDEX:
        vec[_ACTION_TYPE_INDEX[action_type]] = 1.0

    features = action.get("features") or {}
    text = f"{action.get('promptText') or action.get('description') or ''} {action.get('arg') or ''}".lower()

    vec[8] = 1.0 if features.get("is_pass") or ("pass" in text and "disclose" not in text) else 0.0
    vec[9] = 1.0 if features.get("is_claim") or "claim" in text else 0.0
    vec[10] = 1.0 if features.get("is_done") or "done" in text else 0.0
    vec[11] = 1.0 if "cancel" in text else 0.0
    vec[12] = 1.0 if "attack" in text else 0.0
    vec[13] = 1.0 if features.get("is_card") else 0.0
    vec[14] = 1.0 if features.get("is_leader") else 0.0
    vec[15] = 1.0 if features.get("is_base") else 0.0
    vec[16] = 1.0 if features.get("is_unit") else 0.0
    vec[17] = 1.0 if features.get("is_exhausted") else 0.0
    vec[18] = 1.0 if features.get("is_friendly") else 0.0
    vec[19] = 1.0 if action.get("playable", 1.0) else 0.0
    vec[20] = 1.0 if features.get("is_stateful") else 0.0
    vec[21] = 1.0 if features.get("is_macro") else 0.0
    vec[22] = 1.0 if features.get("is_dropdown") else 0.0

    vec[23] = float(features.get("card_power") or 0.0)
    vec[24] = float(features.get("card_hp") or 0.0)
    cost_val = action.get("cost")
    vec[25] = float(cost_val if cost_val is not None else cost) / 10.0

    uuid = str(action.get("uuid") or action.get("cardUuid") or "")
    bucket = 0
    if uuid:
        try:
            if "_" in uuid:
                bucket = int(uuid.split("_")[1]) % 16
            else:
                bucket = int(uuid) % 16
        except (ValueError, TypeError):
            bucket = sum(map(ord, uuid)) % 16
    vec[26 + bucket] = 1.0

    # Button targeting semantics: "to yourself"/"to opponent" choices are
    # otherwise identical menuButtons, which makes the policy coin-flip on
    # effects like "Deal 3 indirect damage to a player".
    vec[42] = 1.0 if (
        features.get("is_self_target")
        or ("yourself" in text and "opponent" not in text)
    ) else 0.0
    vec[43] = 1.0 if (
        features.get("is_opponent_target")
        or ("opponent" in text and "yourself" not in text)
    ) else 0.0
    return vec


def _action_features(action: dict) -> list[float]:
    """Stored feature vector when present, otherwise computed on the fly."""
    stored = action.get("action_features") if isinstance(action, dict) else None
    if isinstance(stored, (list, tuple)) and len(stored) == ACTION_FEATURE_DIM:
        return [float(value) for value in stored]
    return build_action_features(action or {})


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

        # Q(s,a) head: score each candidate action from (state latent, action
        # features) — action identity matters, not its position in the list.
        self.action_head = nn.Sequential(
            nn.Linear(input_size + ACTION_FEATURE_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Legacy positional head: kept so `forward()` / `evaluate()` and any
        # external tooling that introspects `policy_head.weight` keep working.
        # Training no longer uses it (evaluate_q / masked_logits use the Q-head).
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
        temperature: float = 1.0,
        # Backward-compatible no-ops from the old candidate-scoring policy:
        action_feature_size: int = 40,
        action_size: int | None = None,
    ):
        self.device = torch.device(device)
        if action_size is not None:
            max_actions = int(action_size)  # older trainer code passed a fixed action size
        self.obs_size = int(obs_size)
        self.max_actions = int(max_actions)
        self.temperature = max(float(temperature), 1e-3)
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
        Q(s,a) logits: score every candidate action from its feature vector.
        Slots with no candidate / masked actions are forced to -1e9.

        Returns (logits [MAX_ACTIONS], state_value []).
        """
        obs = obs_tensor.to(self.device).float().unsqueeze(0)  # [1, OBS]
        features = self.net.obs_encoder(obs)  # [1, H]
        n = len(available_actions) if available_actions is not None else 0
        n = max(0, min(int(n), self.max_actions))

        logits = torch.full((self.max_actions,), ILLEGAL_LOGIT, dtype=torch.float32, device=self.device)
        value = self.net.value_head(features).squeeze(-1).squeeze(0)
        if n == 0:
            return logits, value

        action_feats = torch.tensor(
            [_action_features(action) for action in available_actions],
            dtype=torch.float32,
            device=self.device,
        )  # [n, F]
        q = self.net.action_head(torch.cat([features.expand(n, -1), action_feats], dim=-1)).squeeze(-1)  # [n]
        logits[:n] = q
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
        # Sampling temperature: >1 keeps exploration alive when logits saturate.
        dist = torch.distributions.Categorical(logits=logits / self.temperature)
        action_tensor = dist.sample()
        return int(action_tensor.item()), dist.log_prob(action_tensor), value.detach()

    def evaluate(self, obs_batch: torch.Tensor, action_indices: torch.Tensor):
        """(log_probs, values) for a batch of stored transitions (training helper)."""
        obs = obs_batch.to(self.device).float()
        logits, values = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        indices = action_indices.to(self.device).long()
        return dist.log_prob(indices), values

    def evaluate_q(self, obs_batch: torch.Tensor, candidate_feats: list, masks: list | None = None):
        """Q-values for a batch of steps under the CURRENT policy.

        - obs_batch: [B, OBS_DIM]
        - candidate_feats: list of B tensors, each [n_i, ACTION_FEATURE_DIM]
          (the candidate set the agent chose among, in slot order)
        - masks: optional list of B legal masks (numpy/int arrays of length max_actions)

        Returns (q_logits [B, MAX_ACTIONS] with ILLEGAL_LOGIT on illegal slots,
                 values [B])."""
        obs = obs_batch.to(self.device).float()
        features = self.net.obs_encoder(obs)  # [B, H]
        batch_size = obs.shape[0]
        q_logits = torch.full((batch_size, self.max_actions), ILLEGAL_LOGIT, dtype=torch.float32, device=self.device)
        values = self.net.value_head(features).squeeze(-1)  # [B]

        repeats = torch.tensor([len(f) for f in candidate_feats], device=self.device)
        if int(repeats.sum()) > 0:
            feat_flat = torch.cat([f.to(self.device).float() for f in candidate_feats], dim=0)
            obs_flat = features.repeat_interleave(repeats, dim=0)  # [M, H]
            q_flat = self.net.action_head(torch.cat([obs_flat, feat_flat], dim=-1)).squeeze(-1)  # [M]
            offsets = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(repeats, dim=0)])
            for i in range(batch_size):
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                if hi > lo:
                    q_logits[i, :hi - lo] = q_flat[lo:hi]

        if masks is not None:
            for i in range(batch_size):
                mask = masks[i]
                if not torch.is_tensor(mask):
                    mask = torch.tensor(mask, device=self.device)
                else:
                    mask = mask.to(self.device)
                q_logits[i] = torch.where(mask > 0.0, q_logits[i], torch.full_like(q_logits[i], ILLEGAL_LOGIT))
        return q_logits, values

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
        n = max(0, min(int(n), self.max_actions))  # guard: indices must fit the mask
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
