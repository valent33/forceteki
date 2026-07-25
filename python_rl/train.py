import argparse
import copy
import json
import math
import os
import random
import re
import time
import torch
import csv
from tqdm import tqdm

from swu_env import SWUEnv
from policy import RandomActionPolicy
from runner import EpisodeLogger
from torch_policy import TorchPolicy
from deck_utils import load_deck


def discounted_returns(rewards, gamma=0.99):
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    if returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def _board_power(state_section: dict | None, key: str) -> float:
    if not state_section:
        return 0.0
    player = state_section.get(key) or {}
    total = 0.0
    for zone in ("spaceArena", "groundArena"):
        for card in player.get(zone, []):
            total += float(card.get("power") or card.get("printedPower") or 0.0)
    return total


def _board_hp(state_section: dict | None, key: str) -> float:
    if not state_section:
        return 0.0
    player = state_section.get(key) or {}
    total = 0.0
    for zone in ("spaceArena", "groundArena"):
        for card in player.get(zone, []):
            total += float(card.get("hp") or card.get("remainingHp") or card.get("currentHp") or 0.0)
    return total


def _player_key_for_id(state: dict | None, player_id: str) -> str | None:
    if not state:
        return None
    if str(state.get("player1Id")) == str(player_id):
        return "player1"
    if str(state.get("player2Id")) == str(player_id):
        return "player2"
    return None


def _unit_board_metrics(state_section: dict | None, key: str) -> dict[str, float]:
    player = (state_section or {}).get(key) or {}
    metrics = {
        "base_hp": 0.0,
        "leader_hp": 0.0,
        "board_power": 0.0,
        "board_hp": 0.0,
        "board_damage": 0.0,
        "unit_count": 0.0,
        "exhausted_count": 0.0,
        "hand_count": float(len(player.get("hand", []))),
        "ready_resources": float(player.get("readyResourceCount") or 0.0),
        "credits": float(player.get("credits") or 0.0),
    }

    base = player.get("base") or {}
    leader = player.get("leader") or {}

    def _max_hp(card: dict | None) -> float:
        if not card:
            return 0.0
        return float(card.get("hp") or card.get("remainingHp") or card.get("currentHp") or card.get("maxHp") or 0.0)

    def _remaining_hp(card: dict | None) -> float:
        """remaining HP = max HP - damage"""
        if not card:
            return 0.0
        max_hp = float(card.get("hp") or card.get("remainingHp") or card.get("currentHp") or card.get("maxHp") or 0.0)
        dmg = float(card.get("damage") or 0.0)
        return max(0.0, max_hp - dmg)

    def _power(card: dict | None) -> float:
        if not card:
            return 0.0
        return float(card.get("power") or card.get("printedPower") or 0.0)

    def _damage(card: dict | None) -> float:
        if not card:
            return 0.0
        return float(card.get("damage") or 0.0)

    metrics["base_hp"] = _remaining_hp(base)
    metrics["leader_hp"] = _remaining_hp(leader)

    for zone in ("spaceArena", "groundArena"):
        for card in player.get(zone, []):
            metrics["unit_count"] += 1.0
            metrics["board_power"] += _power(card)
            metrics["board_hp"] += _remaining_hp(card)
            metrics["board_damage"] += _damage(card)
            if card.get("exhausted") or card.get("isExhausted") or card.get("is_exhausted"):
                metrics["exhausted_count"] += 1.0

    for card in (base, leader):
        metrics["board_power"] += _power(card)
        metrics["board_hp"] += _remaining_hp(card)
        metrics["board_damage"] += _damage(card)

    return metrics


def _is_regroup_phase(phase: object) -> bool:
    return "regroup" in str(phase or "").lower()


def _describe_action(action: dict[str, object], index: int) -> str:
    action_type = str(action.get("actionType", "unknown"))
    label = action.get("promptText") or action.get("internalName") or action.get("uuid") or "unknown"
    return f"[{index}] {action_type}: {label}"


def _log_available_actions(logger: EpisodeLogger, player_label: str, actions: list[dict[str, object]]) -> None:
    if not actions:
        logger.log(f"[{player_label}] available actions: none", player_id=player_label if player_label in {"111", "222"} else None)
        return

    logger.log(f"[{player_label}] available actions:", player_id=player_label if player_label in {"111", "222"} else None)
    for index, action in enumerate(actions):
        logger.log(f"  {_describe_action(action, index)}", player_id=player_label if player_label in {"111", "222"} else None)


def _load_deck_keys(decks_file: str) -> list[str]:
    with open(decks_file, "r", encoding="utf-8") as handle:
        decks_db = json.load(handle)
    if not isinstance(decks_db, dict):
        raise ValueError(f"Expected {decks_file} to contain a JSON object of deck definitions")
    return sorted(str(key) for key in decks_db.keys())


def _sample_episode_decks(deck_keys: list[str]) -> tuple[str, str]:
    if not deck_keys:
        raise ValueError("No deck keys available to sample")
    p1_key = random.choice(deck_keys)
    p2_key = random.choice(deck_keys)
    return p1_key, p2_key


def _build_reset_payload(p1_key: str | None, p2_key: str | None, decks_file: str) -> tuple[dict[str, object], dict[str, object]]:
    reset_options: dict[str, object] = {
        "phase": "setup",
        "player1": {"hasInitiative": True},
    }

    deck_meta: dict[str, object] = {}

    if p1_key:
        leader, base, deck = load_deck(p1_key, decks_file)
        reset_options["p1Leader"] = leader
        reset_options["p1Base"] = base
        reset_options["p1Cards"] = deck
        deck_meta["p1"] = p1_key

    if p2_key:
        leader, base, deck = load_deck(p2_key, decks_file)
        reset_options["p2Leader"] = leader
        reset_options["p2Base"] = base
        reset_options["p2Cards"] = deck
        deck_meta["p2"] = p2_key

    reset_payload = {
        "p1Leader": reset_options.get("p1Leader"),
        "p1Base": reset_options.get("p1Base"),
        "p1Cards": reset_options.get("p1Cards"),
        "p2Leader": reset_options.get("p2Leader"),
        "p2Base": reset_options.get("p2Base"),
        "p2Cards": reset_options.get("p2Cards"),
        "options": {
            "phase": "setup",
            "player1": {"hasInitiative": True},
        },
    }

    return reset_payload, deck_meta


def _load_checkpoint(policy: TorchPolicy, checkpoint_path: str, device: str) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata: dict[str, object] = {}

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        policy.net.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            policy.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        metadata = {key: value for key, value in checkpoint.items() if key not in {"model_state_dict", "optimizer_state_dict"}}
    else:
        policy.net.load_state_dict(checkpoint)

    return metadata


def _infer_episode_from_checkpoint_path(checkpoint_path: str) -> int | None:
    filename = os.path.basename(checkpoint_path)
    match = re.search(r"policy_ep(\d+)\.(?:pt|ckpt)$", filename)
    if match:
        return int(match.group(1))
    return None


class SnapshotOpponent:
    """Opponent that starts random, then switches to a frozen copy of the agent's
    policy that lags behind by `lag` episodes, synced every `sync_interval` episodes."""

    def __init__(self, agent_policy, device, warmup=50, lag=10, verbose=False):
        self.random_policy = RandomActionPolicy()
        self.warmup = warmup
        self.lag = lag
        self.current_ep = 0
        self._using_frozen = False
        self._verbose = verbose

        # Build a detached copy of the agent's network
        obs_size = agent_policy.net.obs_encoder[0].in_features
        self.frozen_policy = TorchPolicy(
            obs_size=obs_size,
            action_feature_size=agent_policy.action_feature_size,
            device=device,
        )

    def sync_from(self, source_policy):
        """Copy the agent's current weights into the frozen opponent network."""
        self.frozen_policy.net.load_state_dict(source_policy.net.state_dict())

    def on_episode_end(self, episode_number: int):
        """Call at the end of each episode to manage warmup / lag."""
        self.current_ep = episode_number
        if not self._using_frozen and episode_number >= self.warmup:
            self._using_frozen = True
            if self._verbose:
                print(f"[opponent] warmup done, switching to frozen policy at episode {episode_number}")

    # def _safe_fallback(self, env, actions: list) -> int | None:
    #     """If the prompt has both card-click actions and a safe "Done"/"Pass"/"Cancel"
    #     button, prefer the button to avoid getting stuck on multi-select prompts."""
    #     has_card_clicks = any(a.get("actionType") in {"clickCard", "displayCardClick", "perCardMenuButton"} for a in actions)
    #     if not has_card_clicks:
    #         return None
    #     safe_texts = {"done", "pass", "take nothing", "choose nothing", "cancel", "play cards in selection order"}
    #     for i, a in enumerate(actions):
    #         if a.get("actionType") == "clickPrompt":
    #             text = str(a.get("promptText", "")).strip().lower()
    #             if text in safe_texts:
    #                 return i
    #     return None

    def choose_action_index(self, env) -> int | None:
        if not self._using_frozen:
            return self.random_policy.choose_action_index(env)

        actions = list(env.available_actions)
        if not actions:
            return None

        # fallback = self._safe_fallback(env, actions)
        # if fallback is not None:
        #     return fallback

        obs_vec = torch.tensor(env._get_obs(), dtype=torch.float32)
        idx, _ = self.frozen_policy.select_action(obs_vec, actions)
        return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", default="http://localhost:3005")
    parser.add_argument("--player_id", default="111")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_dir", default="python_rl/runs/train")
    parser.add_argument("--decks_file", type=str, default="decks.json")
    parser.add_argument("--p1", type=str, help="Deck key for player 1")
    parser.add_argument("--p2", type=str, help="Deck key for player 2")
    parser.add_argument("--randomize_decks", action="store_true", help="Sample fresh decks from decks.json for every episode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from a saved checkpoint (.pt or .ckpt)")
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Write a numbered checkpoint every N episodes")
    parser.add_argument("--stall_polls", type=int, default=40, help="Abort an episode after this many repeated no-action polls in the same prompt state")
    parser.add_argument("--update_every", type=int, default=1, help="Accumulate this many episodes before each policy update (minibatch). 1 = classic REINFORCE, 8-16 recommended for stability")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logs; show a tqdm progress bar instead")
    parser.add_argument("--opponent_policy", type=str, default="random", choices=["random", "old_policy"], help="Policy to use for opponent actions")
    parser.add_argument("--opponent_warmup", type=int, default=50, help="Episodes of random opponent before switching to old_policy")
    parser.add_argument("--opponent_lag", type=int, default=10, help="How many episodes behind the agent the opponent policy snapshot is taken")
    args = parser.parse_args()

    verbose = not args.quiet
    logger = EpisodeLogger(log_dir=args.log_dir, verbose=verbose)
    env = SWUEnv(server_url=args.server_url, player_id=args.player_id, single_agent_mode=True)
    policy = TorchPolicy(obs_size=64, action_feature_size=36, lr=args.lr, device=args.device)

    # Opponent setup — defined here so we can reference the agent's policy
    if args.opponent_policy == "random":
        opponent = RandomActionPolicy()
    else:
        opponent = SnapshotOpponent(policy, args.device, warmup=args.opponent_warmup, lag=args.opponent_lag, verbose=verbose)

    checkpoint_metadata: dict[str, object] = {}
    start_episode = 0
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        checkpoint_metadata = _load_checkpoint(policy, args.checkpoint, args.device)
        if verbose:
            print(f"Loaded checkpoint from {args.checkpoint}")
        if checkpoint_metadata and verbose:
            print(f"Checkpoint metadata: {checkpoint_metadata}")
        start_episode = int(checkpoint_metadata.get("episode") or _infer_episode_from_checkpoint_path(args.checkpoint) or 0)

    deck_keys = _load_deck_keys(args.decks_file) if args.randomize_decks else []

    fixed_reset_payload, fixed_deck_meta = _build_reset_payload(args.p1, args.p2, args.decks_file)

    # Batch accumulators: accumulate episodes before each policy update
    batch_logps: list[torch.Tensor] = []
    batch_returns: list[torch.Tensor] = []

    episode_range = range(args.episodes)
    pbar = tqdm(episode_range, desc="Training", disable=not args.quiet, unit="ep")
    actual_wins: list[float] = []

    for ep in episode_range:
        episode_number = start_episode + ep + 1
        if verbose:
            print(f"=== Episode {episode_number}/{start_episode + args.episodes} ===")
        if args.randomize_decks:
            p1_key, p2_key = _sample_episode_decks(deck_keys)
            reset_payload, deck_meta = _build_reset_payload(p1_key, p2_key, args.decks_file)
        else:
            reset_payload, deck_meta = fixed_reset_payload, fixed_deck_meta

        obs, info = env.reset(options=reset_payload)
        logger.log(
            f"[reset] phase={info.get('phase')} activePlayer={info.get('activePlayer')} valid_actions={info.get('num_valid_actions')} prompts={info.get('activePrompts')}",
            player_id=args.player_id,
        )
        logger.log(
            f"[reset] deck_args p1={deck_meta.get('p1')!r} p2={deck_meta.get('p2')!r} player1Id={env.current_state.get('player1Id') if env.current_state else None} player2Id={env.current_state.get('player2Id') if env.current_state else None}",
            player_id=args.player_id,
        )
        logger.record_rl_transition({"event": "reset", "player_id": args.player_id, "state": env.current_state, "available_actions": env.available_actions, "info": info, "decks": deck_meta})

        logps = []
        rewards = []
        step_idx = 0
        terminated = False

        episode_metrics = {
            "episode": episode_number,
            "agent_turns": 0,
            "opponent_turns": 0,
            "agent_rewards": 0.0,
            "opponent_rewards": 0.0,
            "total_rewards": 0.0,
            "total_reward_steps": 0,
            "valid_actions_sum": 0,
            "valid_actions_count": 0,
            "agent_valid_actions_sum": 0,
            "agent_valid_actions_count": 0,
            "agent_ready_resources_sum": 0.0,
            "agent_credits_sum": 0.0,
            "agent_board_power_sum": 0.0,
            "agent_board_hp_sum": 0.0,
            "agent_board_damage_sum": 0.0,
            "agent_unit_count_sum": 0.0,
            "agent_exhausted_sum": 0.0,
            "agent_base_hp_sum": 0.0,
            "agent_leader_hp_sum": 0.0,
            "opp_board_power_sum": 0.0,
            "opp_board_hp_sum": 0.0,
            "opp_board_damage_sum": 0.0,
            "opp_unit_count_sum": 0.0,
            "opp_exhausted_sum": 0.0,
            "opp_base_hp_sum": 0.0,
            "opp_leader_hp_sum": 0.0,
            "agent_hand_sum": 0,
            "opp_hand_sum": 0,
            "agent_max_valid_actions": 0,
            "final_phase": None,
            "winner": None,
            "regroup_segments": [],
            "regroup_segment_count": 0,
            "regroup_action_total": 0,
            "regroup_card_action_total": 0,
            "regroup_agent_action_total": 0,
            "regroup_opponent_action_total": 0,
        }

        seen_regroup_phase = False
        current_regroup_action_count = 0
        current_regroup_card_action_count = 0
        current_regroup_agent_action_count = 0
        current_regroup_opponent_action_count = 0
        current_regroup_start_step = None
        current_regroup_rewards = 0.0
        current_regroup_steps = 0
        no_action_poll_count = 0
        last_stall_signature = None
        turn_number = 0
        last_turn_phase = None

        def _finalize_regroup_segment(next_phase: str | None) -> None:
            nonlocal current_regroup_action_count
            nonlocal current_regroup_card_action_count
            nonlocal current_regroup_agent_action_count
            nonlocal current_regroup_opponent_action_count
            nonlocal current_regroup_start_step
            nonlocal current_regroup_rewards
            nonlocal current_regroup_steps
            if not seen_regroup_phase or current_regroup_action_count <= 0:
                current_regroup_action_count = 0
                current_regroup_card_action_count = 0
                current_regroup_agent_action_count = 0
                current_regroup_opponent_action_count = 0
                current_regroup_start_step = None
                current_regroup_rewards = 0.0
                current_regroup_steps = 0
                return

            segment = {
                "start_step": current_regroup_start_step,
                "end_step": step_idx,
                "action_count": current_regroup_action_count,
                "card_action_count": current_regroup_card_action_count,
                "agent_action_count": current_regroup_agent_action_count,
                "opponent_action_count": current_regroup_opponent_action_count,
                "reward_total": current_regroup_rewards,
                "reward_per_step": current_regroup_rewards / max(1, current_regroup_steps),
                "next_phase": next_phase,
            }
            episode_metrics["regroup_segments"].append(segment)
            episode_metrics["regroup_segment_count"] += 1
            episode_metrics["regroup_action_total"] += current_regroup_action_count
            episode_metrics["regroup_card_action_total"] += current_regroup_card_action_count
            episode_metrics["regroup_agent_action_total"] += current_regroup_agent_action_count
            episode_metrics["regroup_opponent_action_total"] += current_regroup_opponent_action_count

            current_regroup_action_count = 0
            current_regroup_card_action_count = 0
            current_regroup_agent_action_count = 0
            current_regroup_opponent_action_count = 0
            current_regroup_start_step = None
            current_regroup_rewards = 0.0
            current_regroup_steps = 0

        while not terminated and step_idx < args.max_steps:
            info = env._get_info()
            active = info.get("activePlayer")
            state_snapshot = env.current_state or {}
            prompt_snapshot = state_snapshot.get("prompts") or {}
            phase = str(info.get("phase") or "")
            valid_actions = len(env.available_actions)
            stall_signature = (
                str(active),
                phase,
                valid_actions,
                str((prompt_snapshot.get("player1") or {}).get("menuTitle", "")),
                str((prompt_snapshot.get("player2") or {}).get("menuTitle", "")),
            )
            if valid_actions == 0 and str(active) == str(args.player_id):
                if stall_signature == last_stall_signature:
                    no_action_poll_count += 1
                else:
                    no_action_poll_count = 1
                last_stall_signature = stall_signature
            else:
                no_action_poll_count = 0
                last_stall_signature = None

            episode_metrics["valid_actions_sum"] += valid_actions
            episode_metrics["valid_actions_count"] += 1
            episode_metrics["agent_max_valid_actions"] = max(episode_metrics["agent_max_valid_actions"], valid_actions)
            state_section = state_snapshot.get("state") or {}
            agent_key = _player_key_for_id(state_snapshot, args.player_id)
            opp_key = "player2" if agent_key == "player1" else "player1"
            agent_board = _unit_board_metrics(state_section, agent_key or "player1")
            opp_board = _unit_board_metrics(state_section, opp_key)
            episode_metrics["agent_base_hp_sum"] += agent_board["base_hp"]
            episode_metrics["agent_leader_hp_sum"] += agent_board["leader_hp"]
            episode_metrics["agent_board_power_sum"] += agent_board["board_power"]
            episode_metrics["agent_board_hp_sum"] += agent_board["board_hp"]
            episode_metrics["agent_board_damage_sum"] += agent_board["board_damage"]
            episode_metrics["agent_unit_count_sum"] += agent_board["unit_count"]
            episode_metrics["agent_exhausted_sum"] += agent_board["exhausted_count"]
            episode_metrics["agent_ready_resources_sum"] += agent_board["ready_resources"]
            episode_metrics["agent_credits_sum"] += agent_board["credits"]
            episode_metrics["agent_hand_sum"] += int(agent_board["hand_count"])
            episode_metrics["opp_base_hp_sum"] += opp_board["base_hp"]
            episode_metrics["opp_leader_hp_sum"] += opp_board["leader_hp"]
            episode_metrics["opp_board_power_sum"] += opp_board["board_power"]
            episode_metrics["opp_board_hp_sum"] += opp_board["board_hp"]
            episode_metrics["opp_board_damage_sum"] += opp_board["board_damage"]
            episode_metrics["opp_unit_count_sum"] += opp_board["unit_count"]
            episode_metrics["opp_exhausted_sum"] += opp_board["exhausted_count"]
            episode_metrics["opp_hand_sum"] += int(opp_board["hand_count"])
            logger.log(
                f"[loop] step={step_idx} phase={phase} activePlayer={active} valid_actions={len(env.available_actions)} "
                f"prompt1={str((prompt_snapshot.get('player1') or {}).get('menuTitle', ''))!r} "
                f"prompt2={str((prompt_snapshot.get('player2') or {}).get('menuTitle', ''))!r}",
                player_id=args.player_id,
            )
            _log_available_actions(logger, args.player_id, list(env.available_actions))

            # Detect turn boundaries: a new turn begins when entering regroup phase
            if _is_regroup_phase(phase) and last_turn_phase != "regroup":
                turn_number += 1
            last_turn_phase = phase

            # Record step analysis data
            step_analysis_payload = {
                "episode": episode_number,
                "step_index": step_idx,
                "turn_number": turn_number,
                "active_player_id": str(active),
                "acting_player_id": None,  # Will be filled below
                "phase": phase,
                "valid_actions_count": valid_actions,
                "agent_base_hp": agent_board["base_hp"],
                "agent_leader_hp": agent_board["leader_hp"],
                "agent_board_power": agent_board["board_power"],
                "agent_board_hp": agent_board["board_hp"],
                "agent_board_damage": agent_board["board_damage"],
                "agent_unit_count": agent_board["unit_count"],
                "agent_exhausted_count": agent_board["exhausted_count"],
                "agent_ready_resources": agent_board["ready_resources"],
                "agent_credits": agent_board["credits"],
                "agent_hand_count": agent_board["hand_count"],
                "opp_base_hp": opp_board["base_hp"],
                "opp_leader_hp": opp_board["leader_hp"],
                "opp_board_power": opp_board["board_power"],
                "opp_board_hp": opp_board["board_hp"],
                "opp_board_damage": opp_board["board_damage"],
                "opp_unit_count": opp_board["unit_count"],
                "opp_exhausted_count": opp_board["exhausted_count"],
                "opp_hand_count": opp_board["hand_count"],
                "reward": 0.0,  # Will be filled below
                "terminated": False,  # Will be filled below
                "truncated": False,  # Will be filled below
            }

            if no_action_poll_count >= args.stall_polls and str(active) == str(args.player_id):
                logger.log(
                    f"[warning] Stalled prompt detected after {no_action_poll_count} polls with zero legal actions. "
                    f"Aborting episode to skip bad state. phase={phase} prompt1={str((prompt_snapshot.get('player1') or {}).get('menuTitle', ''))!r} "
                    f"prompt2={str((prompt_snapshot.get('player2') or {}).get('menuTitle', ''))!r}",
                    player_id=args.player_id,
                )
                logger.record_rl_transition({
                    "event": "episode_abort",
                    "reason": "stalled_no_action",
                    "player_id": args.player_id,
                    "step_index": step_idx,
                    "state": state_snapshot,
                    "available_actions": list(env.available_actions),
                    "info": info,
                    "stall_polls": no_action_poll_count,
                })
                episode_metrics["final_phase"] = phase
                terminated = True
                break

            if _is_regroup_phase(phase):
                _finalize_regroup_segment(phase)
                seen_regroup_phase = True
                current_regroup_start_step = step_idx
                # Log turn summary at the start of a new turn (regroup phase entry)
                logger.record_turn_summary({
                    "episode": episode_number,
                    "turn_number": turn_number,
                    "step_index": step_idx,
                    "agent_base_hp": agent_board["base_hp"],
                    "agent_leader_hp": agent_board["leader_hp"],
                    "agent_board_power": agent_board["board_power"],
                    "agent_board_hp": agent_board["board_hp"],
                    "agent_board_damage": agent_board["board_damage"],
                    "agent_unit_count": agent_board["unit_count"],
                    "agent_exhausted_count": agent_board["exhausted_count"],
                    "agent_ready_resources": agent_board["ready_resources"],
                    "agent_credits": agent_board["credits"],
                    "agent_hand_count": agent_board["hand_count"],
                    "opp_base_hp": opp_board["base_hp"],
                    "opp_leader_hp": opp_board["leader_hp"],
                    "opp_board_power": opp_board["board_power"],
                    "opp_board_hp": opp_board["board_hp"],
                    "opp_board_damage": opp_board["board_damage"],
                    "opp_unit_count": opp_board["unit_count"],
                    "opp_exhausted_count": opp_board["exhausted_count"],
                    "opp_hand_count": opp_board["hand_count"],
                })

            if str(active) == str(args.player_id):
                episode_metrics["agent_turns"] += 1
                episode_metrics["agent_valid_actions_sum"] += valid_actions
                episode_metrics["agent_valid_actions_count"] += 1
                obs_vec = torch.tensor(env._get_obs(), dtype=torch.float32)
                available_actions = list(env.available_actions)
                action, logp = policy.select_action(obs_vec, available_actions)
                if action is None:
                    # Refresh before waiting so a stale prompt snapshot does not spin forever.
                    try:
                        env.refresh()
                    except Exception as exc:
                        logger.log(f"[agent] refresh failed while waiting for actions: {exc}", player_id=args.player_id)
                        episode_metrics["final_phase"] = phase
                        terminated = True
                        break

                    if len(env.available_actions) == 0:
                        time.sleep(0.01)
                    else:
                        continue

                    if no_action_poll_count >= args.stall_polls:
                        logger.log(
                            f"[warning] Stalled prompt detected after {no_action_poll_count} polls with zero legal actions. "
                            f"Aborting episode to skip bad state. phase={phase} prompt1={str((prompt_snapshot.get('player1') or {}).get('menuTitle', ''))!r} "
                            f"prompt2={str((prompt_snapshot.get('player2') or {}).get('menuTitle', ''))!r}",
                            player_id=args.player_id,
                        )
                        logger.record_rl_transition({
                            "event": "episode_abort",
                            "reason": "stalled_no_action",
                            "player_id": args.player_id,
                            "step_index": step_idx,
                            "state": state_snapshot,
                            "available_actions": list(env.available_actions),
                            "info": info,
                            "stall_polls": no_action_poll_count,
                        })
                        episode_metrics["final_phase"] = phase
                        terminated = True
                        break
                    continue

                # guard: if action unexpectedly out of range, skip it
                if action >= len(available_actions) or action < 0:
                    # record a warning and skip
                    logger.log(f"Policy produced invalid action {action} for {len(available_actions)} available", player_id=args.player_id)
                    time.sleep(0.01)
                    continue

                chosen_action = available_actions[action]
                logger.log(f"[agent] p1 chose [{action}] {_describe_action(chosen_action, action)}", player_id=args.player_id)

                try:
                    _, reward, terminated, truncated, step_info = env.step(action)
                except Exception as exc:
                    logger.log(
                        f"[agent] step failed; skipping episode: {exc}",
                        player_id=args.player_id,
                    )
                    logger.record_rl_transition({
                        "event": "step_error",
                        "player_id": args.player_id,
                        "step_index": step_idx,
                        "state": state_snapshot,
                        "available_actions": available_actions,
                        "action_index": action,
                        "action": chosen_action,
                        "reward": -10.0,
                        "terminated": True,
                        "truncated": False,
                        "next_state": copy.deepcopy(env.current_state),
                        "info": info,
                        "error": str(exc),
                    })
                    episode_metrics["final_phase"] = phase
                    episode_metrics["agent_rewards"] += -10.0
                    episode_metrics["total_rewards"] += -10.0
                    terminated = True
                    truncated = False
                    step_info = info
                    break
                episode_metrics["total_rewards"] += float(reward)
                episode_metrics["total_reward_steps"] += 1
                episode_metrics["agent_rewards"] += float(reward)
                if not _is_regroup_phase(phase):
                    current_regroup_action_count += 1
                    current_regroup_agent_action_count += 1
                    current_regroup_rewards += float(reward)
                    current_regroup_steps += 1
                    if str(chosen_action.get("actionType") or "") in {"clickCard", "macro_resource_cards"}:
                        current_regroup_card_action_count += 1
                state_section = (step_info or {}).get("state_dict") or env.current_state or {}
                logger.record_rl_transition({
                    "event": "step",
                    "player_id": args.player_id,
                    "step_index": step_idx,
                    "state": state_snapshot,
                    "available_actions": available_actions,
                    "action_index": action,
                    "action": chosen_action,
                    "reward": reward,
                    "terminated": terminated,
                    "info": step_info,
                })
                step_analysis_payload["acting_player_id"] = args.player_id
                step_analysis_payload["reward"] = float(reward)
                step_analysis_payload["terminated"] = terminated
                step_analysis_payload["truncated"] = truncated
                logger.record_step_analysis_data(step_analysis_payload)
                # only append if we have a valid log-prob
                if logp is not None:
                    logps.append(logp)
                    rewards.append(float(reward))
            else:
                episode_metrics["opponent_turns"] += 1
                # opponent acts
                action = opponent.choose_action_index(env)
                if action is None:
                    try:
                        env.refresh()
                    except Exception as exc:
                        logger.log(f"[opponent] refresh failed while waiting for actions: {exc}", player_id=args.player_id)
                        episode_metrics["final_phase"] = phase
                        terminated = True
                        break

                    if len(env.available_actions) == 0:
                        time.sleep(0.01)
                    else:
                        continue

                    if no_action_poll_count >= args.stall_polls:
                        logger.log(
                            f"[warning] Stalled prompt detected after {no_action_poll_count} polls with zero legal actions. "
                            f"Aborting episode to skip bad state. phase={phase} prompt1={str((prompt_snapshot.get('player1') or {}).get('menuTitle', ''))!r} "
                            f"prompt2={str((prompt_snapshot.get('player2') or {}).get('menuTitle', ''))!r}",
                            player_id=args.player_id,
                        )
                        logger.record_rl_transition({
                            "event": "episode_abort",
                            "reason": "stalled_no_action",
                            "player_id": args.player_id,
                            "step_index": step_idx,
                            "state": state_snapshot,
                            "available_actions": list(env.available_actions),
                            "info": info,
                            "stall_polls": no_action_poll_count,
                        })
                        episode_metrics["final_phase"] = phase
                        terminated = True
                        break
                    continue
                opponent_action = list(env.available_actions)[action] if 0 <= action < len(env.available_actions) else None
                if opponent_action is not None:
                    logger.log(f"[opponent] p2 chose [{action}] {_describe_action(opponent_action, action)}", player_id=args.player_id)
                else:
                    logger.log(f"[opponent] p2 chose [{action}] unknown", player_id=args.player_id)
                try:
                    _, reward, terminated, truncated, step_info = env.step(action)
                except Exception as exc:
                    logger.log(
                        f"[opponent] step failed; aborting episode: {exc}",
                        player_id=args.player_id,
                    )
                    logger.record_rl_transition({
                        "event": "step_error",
                        "player_id": "opponent",
                        "step_index": step_idx,
                        "state": state_snapshot,
                        "available_actions": list(env.available_actions),
                        "action_index": action,
                        "action": opponent_action,
                        "reward": -10.0,
                        "terminated": True,
                        "truncated": False,
                        "next_state": copy.deepcopy(env.current_state),
                        "info": info,
                        "error": str(exc),
                    })
                    episode_metrics["final_phase"] = phase
                    episode_metrics["opponent_rewards"] += -10.0
                    episode_metrics["total_rewards"] += -10.0
                    terminated = True
                    truncated = False
                    step_info = info
                    break
                episode_metrics["total_rewards"] += float(reward)
                episode_metrics["total_reward_steps"] += 1
                episode_metrics["opponent_rewards"] += float(reward)
                if not _is_regroup_phase(phase):
                    current_regroup_action_count += 1
                    current_regroup_opponent_action_count += 1
                    current_regroup_rewards += float(reward)
                    current_regroup_steps += 1
                    if str(opponent_action.get("actionType") if opponent_action else "") in {"clickCard", "macro_resource_cards"}:
                        current_regroup_card_action_count += 1 
                logger.record_rl_transition({
                    "event": "step",
                    "player_id": "opponent",
                    "step_index": step_idx,
                    "state": state_snapshot,
                    "available_actions": list(env.available_actions),
                    "action_index": action,
                    "action": opponent_action,
                    "reward": reward,
                    "terminated": terminated,
                    "info": step_info,
                })
                step_analysis_payload["acting_player_id"] = "opponent"
                step_analysis_payload["reward"] = float(reward)
                step_analysis_payload["terminated"] = terminated
                step_analysis_payload["truncated"] = truncated
                logger.record_step_analysis_data(step_analysis_payload)

            step_idx += 1

            if terminated or truncated:
                episode_metrics["final_phase"] = (step_info or {}).get("phase") if isinstance(step_info, dict) else None
                winners = (env.current_state or {}).get("winners", [])
                if winners:
                    episode_metrics["winner"] = winners[0] if len(winners) == 1 else winners

        _finalize_regroup_segment(None)

        # At episode end, accumulate into batch for periodic REINFORCE update
        if len(rewards) > 0:
            returns = discounted_returns(rewards, gamma=args.gamma)
            batch_logps.extend(logps)
            batch_returns.extend(returns)

        do_update = (
            len(batch_logps) > 0
            and (
                (ep + 1) % args.update_every == 0      # reached the batch boundary
                or ep == args.episodes - 1              # last episode of the run
            )
        )
        if do_update:
            loss_val = policy.update(batch_logps, batch_returns)
            n = len(batch_logps)
            if verbose:
                print(f"Batch update after episode {episode_number} ({n} steps across last {min(args.update_every, ep + 1)} eps), loss={loss_val:.6f}")
            batch_logps = []
            batch_returns = []
        else:
            n_skipped = len(rewards) if rewards else 0
            if n_skipped > 0 and verbose:
                print(f"Episode {episode_number} accumulated {n_skipped} steps into batch (next update in {args.update_every - ((ep + 1) % args.update_every)} ep(s))")

        # Opponent lifecycle: warmup tracking + periodic weight sync
        if isinstance(opponent, SnapshotOpponent):
            opponent.on_episode_end(episode_number)
            # Sync agent weights to opponent every `lag` episodes (always sync on warmup end)
            if opponent._using_frozen and (episode_number % args.opponent_lag == 0):
                opponent.sync_from(policy)
                if verbose:
                    print(f"[opponent] synced policy at episode {episode_number}")

        summary = {
            **episode_metrics,
            "steps": step_idx,
            "agent_reward_mean": (episode_metrics["agent_rewards"] / max(1, episode_metrics["agent_turns"])),
            "opponent_reward_mean": (episode_metrics["opponent_rewards"] / max(1, episode_metrics["opponent_turns"])),
            "total_reward_mean": (episode_metrics["total_rewards"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_valid_actions": (episode_metrics["valid_actions_sum"] / max(1, episode_metrics["valid_actions_count"])),
            "avg_agent_valid_actions": (episode_metrics["agent_valid_actions_sum"] / max(1, episode_metrics["agent_valid_actions_count"])),
            "avg_agent_ready_resources": (episode_metrics["agent_ready_resources_sum"] / max(1, episode_metrics["agent_turns"])),
            "avg_agent_credits": (episode_metrics["agent_credits_sum"] / max(1, episode_metrics["agent_turns"])),
            "avg_agent_hand": (episode_metrics["agent_hand_sum"] / max(1, episode_metrics["agent_turns"])),
            "avg_agent_base_hp": (episode_metrics["agent_base_hp_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_opp_base_hp": (episode_metrics["opp_base_hp_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_agent_leader_hp": (episode_metrics["agent_leader_hp_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_opp_leader_hp": (episode_metrics["opp_leader_hp_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_agent_board_power": (episode_metrics["agent_board_power_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_opp_board_power": (episode_metrics["opp_board_power_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_agent_board_hp": (episode_metrics["agent_board_hp_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_opp_board_hp": (episode_metrics["opp_board_hp_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_agent_board_damage": (episode_metrics["agent_board_damage_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_opp_board_damage": (episode_metrics["opp_board_damage_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_agent_unit_count": (episode_metrics["agent_unit_count_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_opp_unit_count": (episode_metrics["opp_unit_count_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_agent_exhausted": (episode_metrics["agent_exhausted_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_opp_exhausted": (episode_metrics["opp_exhausted_sum"] / max(1, episode_metrics["total_reward_steps"])),
            "avg_opp_hand": (episode_metrics["opp_hand_sum"] / max(1, episode_metrics["opponent_turns"])),
        }

        reward_margin = summary["agent_rewards"] - summary["opponent_rewards"]

        if verbose:
            print(
                f"[episode {episode_number}] steps={summary['steps']} agent_turns={summary['agent_turns']} "
                f"opp_turns={summary['opponent_turns']} agent_reward={summary['agent_rewards']:.3f} total_reward={summary['total_rewards']:.3f} "
                f"avg_valid_actions={summary['avg_valid_actions']:.2f} avg_agent_valid_actions={summary['avg_agent_valid_actions']:.2f} "
                f"avg_agent_base_hp={summary['avg_agent_base_hp']:.2f} avg_opp_base_hp={summary['avg_opp_base_hp']:.2f} "
                f"avg_agent_board_power={summary['avg_agent_board_power']:.2f} avg_opp_board_power={summary['avg_opp_board_power']:.2f} "
                f"avg_agent_board_hp={summary['avg_agent_board_hp']:.2f} avg_opp_board_hp={summary['avg_opp_board_hp']:.2f} "
                f"avg_agent_board_damage={summary['avg_agent_board_damage']:.2f} avg_opp_board_damage={summary['avg_opp_board_damage']:.2f} "
                f"avg_agent_unit_count={summary['avg_agent_unit_count']:.2f} avg_opp_unit_count={summary['avg_opp_unit_count']:.2f} "
                f"regroup_segments={summary['regroup_segment_count']} regroup_actions={summary['regroup_action_total']} "
                f"regroup_card_actions={summary['regroup_card_action_total']} winner={summary['winner']}"
            )
        else:
            if not hasattr(pbar, "_rs"):
                pbar._rs = 0.0
                pbar._rn = 0
            pbar._rs += reward_margin
            pbar._rn += 1
            rolling_rwd = pbar._rs / pbar._rn
            pbar.set_description(f"RwØ {rolling_rwd:+.3f} | Steps {summary['steps']}")
            pbar.update(1)

        logger.record_episode_summary(summary)

        latest_payload = {
            "model_state_dict": policy.net.state_dict(),
            "optimizer_state_dict": policy.optimizer.state_dict(),
            "episode": episode_number,
            "decks": deck_meta,
            "checkpoint_source": args.checkpoint,
        }

        torch.save(policy.net.state_dict(), f"{args.log_dir}/policy_latest.pt")
        torch.save(latest_payload, f"{args.log_dir}/policy_latest.ckpt")

        if args.checkpoint_every > 0 and (episode_number % args.checkpoint_every == 0):
            torch.save(policy.net.state_dict(), f"{args.log_dir}/policy_ep{episode_number}.pt")
            torch.save(
                latest_payload,
                f"{args.log_dir}/policy_ep{episode_number}.ckpt",
            )

    logger.close()


if __name__ == "__main__":
    main()
