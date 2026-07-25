from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stable_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _bucketed_hash_features(text: str, size: int) -> list[float]:
    values = [0.0] * size
    if not text:
        return values
    hashed = _stable_hash(text)
    values[hashed % size] = 1.0
    return values


class CandidateScoringNetwork(nn.Module):
    def __init__(self, obs_size: int = 64, action_feature_size: int = 36, hidden: int = 256):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_feature_size, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden + hidden // 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_batch: torch.Tensor, action_feature_batch: torch.Tensor) -> torch.Tensor:
        obs_embed = self.obs_encoder(obs_batch)
        action_embed = self.action_encoder(action_feature_batch)
        joint = torch.cat([obs_embed, action_embed], dim=-1)
        return self.scorer(joint).squeeze(-1)


class TorchPolicy:
    # How to determine obs_size:
    #   Count every float feature extracted in env._get_obs().
    #   Current encoder fills indices 0-17 (18 features: phase, base/leader HP/power
    #   for both players, hand/board/resource counts) and 21-35 (15 features: normalized
    #   unit totals, power, HP, advantage deltas) = 33 meaningful features.
    #   Set obs_size to the next power of two above your count to give the network
    #   enough capacity while avoiding dead parameters (e.g. 64 for ~33 features).
    #   Verify by adding a print(len(obs)) assertion in _get_obs at runtime.
    def __init__(
        self,
        obs_size: int = 64,
        action_feature_size: int = 36,
        lr: float = 1e-3,
        device: str = "cpu",
        action_size: int | None = None,
    ):
        self.device = torch.device(device)
        if action_size is not None:
            # Backward-compatible no-op: older trainer code passed a fixed action_size.
            # Candidate scoring does not use a global action head.
            action_feature_size = action_feature_size
        self.net = CandidateScoringNetwork(obs_size=obs_size, action_feature_size=action_feature_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.action_feature_size = action_feature_size

    def encode_action(self, action: dict[str, Any] | None) -> list[float]:
        features = [0.0] * self.action_feature_size
        if not action:
            return features

        action_type = str(action.get("actionType", "unknown"))
        type_index = {
            "clickPrompt": 0,
            "clickCard": 1,
            "macro_resource_cards": 2,
        }.get(action_type, 3)
        features[type_index] = 1.0

        # Structured numeric fields (12 slots)
        structured = action.get("features") or {}
        # order: is_stateful,is_macro,is_dropdown,is_done,is_card,is_friendly,is_leader,is_base,is_exhausted,is_unit,card_power,card_hp
        struct_vals = [
            float(structured.get("is_stateful", 0.0)),
            float(structured.get("is_macro", 0.0)),
            float(structured.get("is_dropdown", 0.0)),
            float(structured.get("is_done", 0.0)),
            float(structured.get("is_card", 0.0)),
            float(structured.get("is_friendly", 0.0)),
            float(structured.get("is_leader", 0.0)),
            float(structured.get("is_base", 0.0)),
            float(structured.get("is_exhausted", 0.0)),
            float(structured.get("is_unit", 0.0)),
            float(structured.get("card_power", 0.0)),
            float(structured.get("card_hp", 0.0)),
        ]

        offset = 4
        for i, v in enumerate(struct_vals):
            features[offset + i] = v
        offset += 12

        # A few sparse, stable hash buckets for prompt/card identity.
        prompt_text = str(action.get("promptText", ""))
        internal_name = str(action.get("internalName", ""))
        uuid_text = str(action.get("uuid", ""))

        prompt_hash = _bucketed_hash_features(prompt_text, 8)
        name_hash = _bucketed_hash_features(internal_name, 8)
        uuid_hash = _bucketed_hash_features(uuid_text, 4)

        for idx, value in enumerate(prompt_hash):
            features[offset + idx] = value
        offset += 8
        for idx, value in enumerate(name_hash):
            features[offset + idx] = value
        offset += 8
        for idx, value in enumerate(uuid_hash):
            features[offset + idx] = value

        return features

    def select_action(self, obs_tensor: torch.Tensor, available_actions: list[dict[str, Any]]):
        if not available_actions:
            return None, None

        obs = obs_tensor.to(self.device).float().unsqueeze(0)
        obs_batch = obs.repeat(len(available_actions), 1)
        action_features = torch.tensor(
            [self.encode_action(action) for action in available_actions],
            dtype=torch.float32,
            device=self.device,
        )
        scores = self.net(obs_batch, action_features)
        probs = F.softmax(scores, dim=-1)
        probs = probs / (probs.sum() + 1e-12)
        dist = torch.distributions.Categorical(probs)
        action_tensor = dist.sample()
        action_index = int(action_tensor.item())
        logp = dist.log_prob(action_tensor)
        return action_index, logp

    def update(self, logps, returns):
        if not logps or len(returns) == 0:
            return 0.0

        loss = torch.tensor(0.0, device=self.device)
        for lp, ret in zip(logps, returns):
            loss = loss + (-lp * ret.to(self.device))
        loss = loss / max(1, len(returns))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())