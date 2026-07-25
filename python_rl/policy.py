import json
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyDecision:
    action_index: int
    action_type: str
    description: str


class RandomActionPolicy:
    def _candidate_indices(self, env) -> list[int]:
        actions = getattr(env, "available_actions", None) or []
        if not actions:
            return []

        active_player = getattr(env, "active_player", None)
        if active_player is None:
            return list(range(len(actions)))

        matching = [index for index, action in enumerate(actions) if str(action.get("playerId")) == str(active_player)]
        return matching or list(range(len(actions)))

    def _preferred_fallback(self, env, candidate_indices: list[int]) -> int | None:
        """If there's a mix of card clicks and cancel/done buttons, prefer the button
        to avoid unreliable card-click interactions (e.g. Maz Kanata / Owen Lars prompts
        where the server rejects card clicks after pipeline advances)."""
        actions = getattr(env, "available_actions", None) or []
        has_card_clicks = any(actions[i].get("actionType") == "clickCard" for i in candidate_indices)

        if not has_card_clicks:
            return None  # no card clicks to avoid

        # Look for a safe fallback button
        safe_texts = {"take nothing", "choose nothing", "done", "pass", "cancel"}
        safe_indices = [
            i for i in candidate_indices
            if actions[i].get("actionType") == "clickPrompt"
            and str(actions[i].get("promptText", "")).strip().lower() in safe_texts
        ]
        if safe_indices:
            return random.choice(safe_indices)
        return None

    def choose_action_index(self, env) -> int | None:
        candidate_indices = self._candidate_indices(env)
        if not candidate_indices:
            return None

        # Prefer a safe fallback button when card clicks are present (server-side crash avoidance)
        fallback = self._preferred_fallback(env, candidate_indices)
        if fallback is not None:
            return fallback

        return random.choice(candidate_indices)

    def describe_choice(self, env, action_index: int) -> PolicyDecision:
        action = env.available_actions[action_index]
        action_type = action.get("actionType", "unknown")
        description = action.get("internalName", action.get("promptText", action.get("uuid", "unknown")))
        return PolicyDecision(action_index=action_index, action_type=action_type, description=str(description))


class HeuristicActionPolicy(RandomActionPolicy):
    def choose_action_index(self, env) -> int | None:
        candidate_indices = self._candidate_indices(env)
        if not candidate_indices:
            return None

        for index in candidate_indices:
            action = env.available_actions[index]
            prompt_text = str(action.get("promptText", "")).strip().lower()
            action_type = action.get("actionType")
            if action_type == "macro_resource_cards":
                return index
            if action_type == "clickPrompt" and prompt_text in {"resource", "keep", "claim initiative", "done", "pass"}:
                return index

        return super().choose_action_index(env)


def format_state_brief(state: dict[str, Any] | None) -> str:
    if not state:
        return "<no state>"

    phase = state.get("phase", "unknown")
    active_player = state.get("activePlayer", "unknown")
    prompts = state.get("prompts") or {}
    prompt_bits = []
    for key in ("player1", "player2"):
        prompt = prompts.get(key) or {}
        title = prompt.get("menuTitle") or prompt.get("title") or ""
        if title:
            prompt_bits.append(f"{key}:{title}")

    prompt_text = " | ".join(prompt_bits) if prompt_bits else "no prompts"
    return f"phase={phase}, activePlayer={active_player}, {prompt_text}"


def dump_json_line(prefix: str, payload: dict[str, Any]) -> str:
    return f"{prefix} {json.dumps(payload, default=str)}"