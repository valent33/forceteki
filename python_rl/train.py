"""
Gated Champion/Candidate self-play training loop for the SWU RL environment.

- Training opponents: 80% of episodes use the current champion checkpoint
  (`policy_champion.ckpt`); 20% use a randomly sampled past checkpoint from the
  `checkpoints/` history folder.
- Every `--tournament_every` episodes, an evaluation tournament runs between
  the live candidate network and the champion (`--tournament_games` games,
  first/second player seats alternated evenly). A candidate with win rate >
  `--promote_win_rate` is promoted to `policy_champion.ckpt`.
- A2C loss:  Loss = L_policy + value_coef * L_value - entropy_coef * H_entropy
  computed from `batch_logps`, `batch_values`, `batch_returns`.
- Diagnostics: per-step action dumps are suppressed by default; a clean summary
  line is printed every `--diagnostics_every` episodes; TensorBoard logs are
  written when the `tensorboard` package is available.
"""

import argparse
import copy
import json
import os
import queue
import random
import re
import time
from typing import Any, Callable

import numpy as np
import torch

from swu_env import SWUEnv
from policy import RandomActionPolicy
from runner import EpisodeLogger
from torch_policy import TorchPolicy, build_action_features, _action_features
from deck_utils import load_deck

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:  # tensorboard is an optional dependency
    TENSORBOARD_AVAILABLE = False


CHAMPION_FILENAME = "policy_champion.ckpt"
HISTORY_DIRNAME = "checkpoints"


# ── Reward / state helpers ───────────────────────────────────────────────────
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


def _slim_info(info: dict | None) -> dict | None:
    """Strip the nested full-state copy (`state_dict`) from env info before
    writing transitions to disk — the caller already stores the state snapshot,
    so keeping it twice bloats transitions.jsonl massively."""
    if not isinstance(info, dict):
        return info
    return {key: value for key, value in info.items() if key != "state_dict"}


def _slim_actions(actions, cap: int = 24) -> dict:
    """Compact serialisation of an action list for transitions.jsonl.

    Giant prompts (e.g. "Choose an option from the list" listing every card
    title) used to dump 1700+ full action dicts per transition. Log a count
    plus the first `cap` slimmed actions — enough to debug, small enough to
    keep the log usable."""
    actions = list(actions or [])
    shown = []
    for action in actions[:cap]:
        shown.append({
            key: action.get(key)
            for key in ("actionType", "arg", "uuid", "method", "internalName", "promptText", "cardUuid", "uuids")
            if action.get(key) is not None
        })
    return {"count": len(actions), "shown": shown}


_BUTTON_ARGS = {"cancel", "pass", "done"}


def _bump_action_metrics(metrics: dict, action: dict | None, prompt_title: str = "") -> None:
    """Count strategic clicks per episode for the console summary.

    In headless mode an attack is NOT a button click: the player clicks their
    unit in the action window, then clicks an enemy card under the
    "Choose a target for attack" prompt (InitiateAttackAction). So attacks are
    counted as clickCard actions on enemy targets while an attack-titled
    prompt is open."""
    if not isinstance(action, dict):
        return
    action_type = action.get("actionType")
    title = str(prompt_title or "").lower()
    if action_type == "clickPrompt":
        arg = str(action.get("arg") or "").strip().lower()
        text = str(action.get("promptText") or "").strip().lower()
        # Buttons often carry numeric args (e.g. Cancel has arg=1 on the
        # credit-payment prompt); match on the button text too so no-op
        # cancels are counted instead of vanishing from the summary.
        if arg in _BUTTON_ARGS:
            key = f"{arg}_clicks"
        elif "cancel" in text:
            key = "cancel_clicks"
        elif "pass" in text:
            key = "pass_clicks"
        elif "done" in text:
            key = "done_clicks"
        else:
            key = None
        if key:
            metrics[key] = metrics.get(key, 0) + 1
        if arg == "attack" or "attack" in text:
            metrics["attack_clicks"] = metrics.get("attack_clicks", 0) + 1
    elif action_type == "clickCard" and "attack" in title:
        # The attacker click happens in the action window (no "attack" in its
        # title); the defender/base click under the attack prompt is the attack.
        if (action.get("meta") or {}).get("targetOwner") == "enemy":
            metrics["attack_clicks"] = metrics.get("attack_clicks", 0) + 1


def _log_no_progress(logger, actor: str, step_idx: int, action: dict, env) -> None:
    """Record a step the server did not accept, with the server's own payload,
    so mismatches can be reviewed later (see inspect_server.py)."""
    state = env.current_state or {}
    p_id = str(action.get("playerId") or "")
    p_key = "player1" if str(state.get("player1Id")) == p_id else "player2"
    prompt = (state.get("prompts") or {}).get(p_key) or {}
    payload = {
        "menuTitle": prompt.get("menuTitle"),
        "buttons": [
            {"text": b.get("text"), "arg": b.get("arg"), "disabled": b.get("disabled")}
            for b in (prompt.get("buttons") or [])
        ],
        "selectableCards": prompt.get("selectableCards"),
        "selectedCards": prompt.get("selectedCards"),
        "dropdownListOptions": prompt.get("dropdownListOptions"),
    }
    logger.log(
        f"[no-progress] step={step_idx} actor={actor} prompt={str(prompt.get('menuTitle'))!r} "
        f"action={_describe_action(action, None)} selectable={len(prompt.get('selectableCards') or [])}",
        player_id=actor if actor in {"111", "222"} else None,
    )
    logger.record_rl_transition({
        "event": "no_progress_step",
        "player_id": actor,
        "step_index": step_idx,
        "action": action,
        "server_payload": payload,
        "available_actions": _slim_actions(env.available_actions),
    })


def _snapshot_policy(policy: TorchPolicy, device: str) -> TorchPolicy:
    """Clone the candidate's current weights for self-play (detached copy)."""
    snapshot = TorchPolicy(obs_size=policy.obs_size, max_actions=policy.max_actions, device=device, temperature=policy.temperature)
    snapshot.net.load_state_dict({key: value.detach().clone() for key, value in policy.net.state_dict().items()})
    snapshot.net.eval()
    return snapshot


# ── Gated champion pool ──────────────────────────────────────────────────────
class ChampionPool:
    """
    Persistent champion checkpoint plus a library of historical checkpoints.

    `sample_opponent()` implements the gate:
      80% of episodes → the current champion weights
      20% of episodes → a randomly selected past checkpoint from `checkpoints/`
    """

    def __init__(
        self,
        log_dir: str,
        champion_path: str,
        obs_size: int,
        max_actions: int,
        device: str,
        champion_probability: float = 0.8,
        self_play_probability: float = 0.5,
        verbose: bool = True,
    ):
        self.log_dir = log_dir
        self.champion_path = champion_path
        self.history_dir = os.path.join(log_dir, HISTORY_DIRNAME)
        self.obs_size = obs_size
        self.max_actions = max_actions
        self.device = device
        self.champion_probability = float(champion_probability)
        self.self_play_probability = float(self_play_probability)
        self.verbose = verbose
        os.makedirs(self.history_dir, exist_ok=True)

        self.champion = self._make_policy()
        self.has_champion = False
        if os.path.exists(self.champion_path):
            state_dict = self._load_model_weights(self.champion_path)
            if state_dict is not None:
                self.champion.net.load_state_dict(state_dict)
                self.has_champion = True

    def _make_policy(self) -> TorchPolicy:
        return TorchPolicy(obs_size=self.obs_size, max_actions=self.max_actions, device=self.device)

    @staticmethod
    def _load_model_weights(path: str):
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception:
            return None
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict):
            return checkpoint
        return None

    def save_champion(self, policy: TorchPolicy, episode: int | None = None, reason: str = "") -> str:
        """Persist the candidate's weights as the new champion."""
        payload = {
            "model_state_dict": {key: value.detach().cpu() for key, value in policy.net.state_dict().items()},
            "episode": episode,
            "reason": reason,
        }
        torch.save(payload, self.champion_path)
        self.champion.net.load_state_dict(policy.net.state_dict())
        self.has_champion = True
        if self.verbose:
            print(f"[champion] saved to {self.champion_path}" + (f" — {reason}" if reason else ""))
        return self.champion_path

    def register_checkpoint(self, policy: TorchPolicy, episode: int) -> str:
        """Archive the current weights into the history folder for later sampling."""
        path = os.path.join(self.history_dir, f"policy_ep{episode}.pt")
        torch.save(policy.net.state_dict(), path)
        return path

    def sample_opponent(self, candidate: TorchPolicy | None = None) -> tuple[TorchPolicy | None, str]:
        """Sample an opponent policy. Returns (policy, source).

        Sources: 'self' (snapshot of the candidate for self-play), 'champion',
        'history', or 'random' (no usable pool). Self-play matters: training
        against a frozen pass-bot champion means the agent never experiences
        combat or the terminal win/loss signal, so it collapses into passing."""
        if candidate is not None and random.random() < self.self_play_probability:
            return _snapshot_policy(candidate, self.device), "self"

        if self.has_champion and random.random() < self.champion_probability:
            return self.champion, "champion"

        history = sorted(
            name for name in os.listdir(self.history_dir)
            if name.endswith(".pt") or name.endswith(".ckpt")
        )
        if history:
            state_dict = self._load_model_weights(os.path.join(self.history_dir, random.choice(history)))
            if state_dict is not None:
                snapshot = self._make_policy()
                try:
                    snapshot.net.load_state_dict(state_dict)
                    return snapshot, "history"
                except Exception:
                    pass

        if self.has_champion:
            return self.champion, "champion"
        return None, "random"


# ── Opponent / action helpers ────────────────────────────────────────────────
def _policy_action(policy, env) -> int | None:
    """Sample a masked action index from a TorchPolicy (or a legacy policy)."""
    actions = list(env.available_actions)
    if not actions:
        return None
    if isinstance(policy, TorchPolicy):
        try:
            with torch.no_grad():
                obs_vec = torch.tensor(env._get_obs(), dtype=torch.float32)
                idx, _, _ = policy.select_action(obs_vec, actions, getattr(env, "legal_action_mask", None))
            return idx if idx is not None and 0 <= idx < len(actions) else None
        except Exception:
            return None
    try:
        return policy.choose_action_index(env)
    except Exception:
        return None


class PolicyOpponent:
    """Wraps a frozen TorchPolicy with a random-policy fallback."""

    def __init__(self, policy=None, fallback=None):
        self.policy = policy
        self.fallback = fallback if fallback is not None else RandomActionPolicy()

    def choose_action_index(self, env) -> int | None:
        if self.policy is None:
            return self.fallback.choose_action_index(env)
        index = _policy_action(self.policy, env)
        return index if index is not None else self.fallback.choose_action_index(env)


# ── Evaluation tournament ────────────────────────────────────────────────────
def _resolve_winner(env) -> str | None:
    """'player1' | 'player2' | 'draw' | None (unresolved), from base HP."""
    state = env.current_state or {}
    section = state.get("state") or {}
    p1 = _unit_board_metrics(section, "player1")
    p2 = _unit_board_metrics(section, "player2")
    p1_dead = p1["base_hp"] <= 0.0
    p2_dead = p2["base_hp"] <= 0.0
    if p1_dead and not p2_dead:
        return "player2"
    if p2_dead and not p1_dead:
        return "player1"
    if p1_dead and p2_dead:
        return "draw"
    return None


def play_one_game(env, p1_policy, p2_policy, reset_payload: dict, max_steps: int = 1000, stall_polls: int = 40) -> str | None:
    """Run one full episode to completion. Returns winner seat or None."""
    try:
        env.reset(options=reset_payload)
    except Exception as exc:
        print(f"  [tournament] reset failed: {exc}")
        return None
    last_signature = None
    stall_count = 0

    for _ in range(max_steps):
        info = env._get_info()
        state = env.current_state or {}
        valid_actions = len(env.available_actions)
        prompts = state.get("prompts") or {}
        signature = (
            str(info.get("activePlayer")),
            str(info.get("phase")),
            valid_actions,
            str((prompts.get("player1") or {}).get("menuTitle", "")),
            str((prompts.get("player2") or {}).get("menuTitle", "")),
        )

        if valid_actions == 0:
            stall_count = stall_count + 1 if signature == last_signature else 1
            last_signature = signature
            if stall_count >= stall_polls:
                return None
            try:
                env.refresh()
            except Exception:
                return None
            # time.sleep(0.01)
            continue
        last_signature = signature
        stall_count = 0

        active = str(info.get("activePlayer") or "")
        seat = "player2" if active == str(state.get("player2Id")) else "player1"
        policy = p1_policy if seat == "player1" else p2_policy
        action_index = _policy_action(policy, env)
        if action_index is None:
            try:
                env.refresh()
            except Exception:
                return None
            # time.sleep(0.01)
            continue

        try:
            _, _, terminated, truncated, _ = env.step(action_index)
        except Exception as exc:
            print(f"  [tournament] step failed: {exc}")
            return None
        if terminated or truncated:
            break

    return _resolve_winner(env)


def run_tournament(
    env,
    candidate,
    champion,
    reset_payload_factory: Callable[[], dict],
    games: int = 100,
    max_steps: int = 500,
    stall_polls: int = 40,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Candidate-vs-champion evaluation tournament.

    Player seats alternate evenly (candidate plays P1 on even game indexes),
    so a `games=50` tournament is 25 games per seat. Win rate is computed over
    decisive games (candidate wins + champion wins).
    """
    stats = {
        "games": 0,
        "candidate_wins": 0,
        "champion_wins": 0,
        "draws": 0,
        "unresolved": 0,
        "candidate_win_rate": 0.0,
    }
    for game in range(games):
        candidate_is_p1 = game % 2 == 0
        p1 = candidate if candidate_is_p1 else champion
        p2 = champion if candidate_is_p1 else candidate
        winner = play_one_game(env, p1, p2, reset_payload_factory(), max_steps=max_steps, stall_polls=stall_polls)
        stats["games"] += 1
        if winner is None:
            stats["unresolved"] += 1
        elif winner == "draw":
            stats["draws"] += 1
        elif winner == "player1":
            if candidate_is_p1:
                stats["candidate_wins"] += 1
            else:
                stats["champion_wins"] += 1
        else:
            if candidate_is_p1:
                stats["champion_wins"] += 1
            else:
                stats["candidate_wins"] += 1
        if verbose and (game + 1) % 10 == 0:
            print(f"  [tournament] game {game + 1}/{games} — candidate {stats['candidate_wins']} : "
                  f"champion {stats['champion_wins']} (draws {stats['draws']}, unresolved {stats['unresolved']})")

    decisive = max(1, stats["candidate_wins"] + stats["champion_wins"])
    stats["candidate_win_rate"] = stats["candidate_wins"] / decisive
    return stats


# ── TensorBoard ──────────────────────────────────────────────────────────────
class MetricsBoard:
    """Thin TensorBoard wrapper; silently no-ops when unavailable or disabled."""

    def __init__(self, enabled: bool, log_dir: str):
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard")) if enabled else None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


# ── System perf reporting ────────────────────────────────────────────────────
def _collect_perf() -> dict:
    """CPU/RAM of the training process and every running env server process."""
    perf = {"psutil": False}
    try:
        import psutil
    except ImportError:
        return perf
    perf["psutil"] = True
    me = psutil.Process()
    perf["python_rss_mb"] = round(me.memory_info().rss / 1e6, 1)
    try:
        perf["python_cpu_pct"] = round(me.cpu_percent(interval=0.0), 1)
    except Exception:
        perf["python_cpu_pct"] = None
    servers = []
    for proc in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "envServer.js" in cmd:
                rss = proc.info.get("memory_info")
                servers.append({"pid": proc.info.get("pid"), "rss_mb": round((rss.rss if rss else 0) / 1e6, 1)})
        except Exception:
            continue
    perf["env_servers"] = servers
    return perf


def _collect_gpu() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return " gpu: " + out.stdout.strip().replace("\n", " ")
    except Exception:
        pass
    return ""


def _perf_line(window_start: float, window_steps: int, window_len: int, episode_number: int) -> tuple[float, int]:
    wall = max(1e-6, time.time() - window_start)
    line = (
        f"[perf] episode {episode_number} | last {window_len} eps in {wall:.1f}s "
        f"({window_len / wall:.2f} eps/s, {window_steps / wall:.1f} steps/s)"
    )
    perf = _collect_perf()
    if perf.get("psutil"):
        line += f" | python rss={perf['python_rss_mb']}MB cpu={perf['python_cpu_pct']}%"
        srv = perf.get("env_servers") or []
        if srv:
            line += " | servers: " + ", ".join(f"{s['pid']}({s['rss_mb']}MB)" for s in srv)
    line += _collect_gpu()
    print(line)
    return time.time(), 0


# ── Parallel training (one worker per env server) ───────────────────────────
def _port_from_url(server_url: str) -> int:
    try:
        return int(server_url.rsplit(":", 1)[1].split("/")[0])
    except Exception:
        return 3005


def _atomic_save_state_dict(path: str, state_dict: dict) -> None:
    tmp = path + ".tmp"
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


def _load_state_dict_any(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _section_base_hp(section: dict | None, key: str) -> float:
    """Remaining base HP for a seat. A destroyed base serializes as null —
    that means 0 HP, not a default. Reading it as 30 made every game count as
    unresolved (flat episode/win, deck report with zero wins)."""
    try:
        base = (section or {}).get(key, {}).get("base") or {}
        return float(base.get("hp") or base.get("remainingHp") or base.get("currentHp") or base.get("maxHp") or 0.0)
    except Exception:
        return 0.0


def _run_worker_episode(env, policy, opponent_policy, player_id, reset_payload, max_steps,
                        p1_key=None, p2_key=None, opponent_source="champion"):
    """Run one full episode.

    Returns (obs_list, actions, feature_list, mask_list, rewards, steps,
    winner, agent_hp, opp_hp, p1_key, p2_key, metrics) for the AGENT's steps
    only. `metrics` mirrors the serial loop's per-episode summary so the
    parallel path can write the same episode_summaries.csv schema (dashboards
    read `opponent_rewards`, `agent_turns`, `end_by`, ... from it).
    Log-probs and values are recomputed by the main process under the current
    policy (async-A2C correction), which also avoids pickling grad-enabled
    tensors. `p1_key` is the agent's deck key, `p2_key` the opponent's."""
    obs_list: list = []
    actions: list = []
    feature_list: list = []
    mask_list: list = []
    rewards: list = []
    metrics: dict = {
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
        "agent_max_valid_actions": 0,
        "agent_base_hp_sum": 0.0,
        "agent_leader_hp_sum": 0.0,
        "agent_board_power_sum": 0.0,
        "agent_board_hp_sum": 0.0,
        "agent_board_damage_sum": 0.0,
        "agent_unit_count_sum": 0.0,
        "agent_exhausted_sum": 0.0,
        "agent_ready_resources_sum": 0.0,
        "agent_credits_sum": 0.0,
        "agent_hand_sum": 0.0,
        "opp_base_hp_sum": 0.0,
        "opp_leader_hp_sum": 0.0,
        "opp_board_power_sum": 0.0,
        "opp_board_hp_sum": 0.0,
        "opp_board_damage_sum": 0.0,
        "opp_unit_count_sum": 0.0,
        "opp_exhausted_sum": 0.0,
        "opp_hand_sum": 0.0,
        "cards_played": 0,
        "agent_cards_played": 0,
        "cancel_clicks": 0,
        "pass_clicks": 0,
        "done_clicks": 0,
        "attack_clicks": 0,
        "final_phase": None,
        "ended_by": "max_steps",
        "opponent_source": opponent_source,
    }
    try:
        _, info = env.reset(options=reset_payload)
    except Exception:
        return None
    terminated = False
    step = 0
    while not terminated and step < max_steps:
        step += 1
        active = str(info.get("activePlayer") or "")
        if str(active) == str(player_id):
            actor, collect = policy, True
        else:
            actor, collect = opponent_policy, False

        # Per-episode accounting (mirrors the serial training loop so the
        # parallel path can write the same episode_summaries.csv schema).
        state_now = env.current_state or {}
        seat_key = _player_key_for_id(state_now, player_id) or "player1"
        opp_seat_key = "player2" if seat_key == "player1" else "player1"
        valid_now = len(env.available_actions)
        metrics["valid_actions_sum"] += valid_now
        metrics["valid_actions_count"] += 1
        metrics["agent_max_valid_actions"] = max(metrics["agent_max_valid_actions"], valid_now)
        board_agent = _unit_board_metrics(state_now.get("state") or {}, seat_key)
        board_opp = _unit_board_metrics(state_now.get("state") or {}, opp_seat_key)
        if collect:
            metrics["agent_valid_actions_sum"] += valid_now
            metrics["agent_valid_actions_count"] += 1
        for prefix, board in (("agent_", board_agent), ("opp_", board_opp)):
            metrics[f"{prefix}base_hp_sum"] += board["base_hp"]
            metrics[f"{prefix}leader_hp_sum"] += board["leader_hp"]
            metrics[f"{prefix}board_power_sum"] += board["board_power"]
            metrics[f"{prefix}board_hp_sum"] += board["board_hp"]
            metrics[f"{prefix}board_damage_sum"] += board["board_damage"]
            metrics[f"{prefix}unit_count_sum"] += board["unit_count"]
            metrics[f"{prefix}exhausted_sum"] += board["exhausted_count"]
            metrics[f"{prefix}hand_sum"] += board["hand_count"]
        metrics["agent_ready_resources_sum"] += board_agent["ready_resources"]
        metrics["agent_credits_sum"] += board_agent["credits"]

        if not env.available_actions:
            try:
                env.refresh()
            except Exception:
                metrics["ended_by"] = "refresh_failed"
                break
            continue
        obs_vec = torch.tensor(env._get_obs(), dtype=torch.float32)
        try:
            action, logp, value = actor.select_action(obs_vec, list(env.available_actions), env.legal_action_mask)
        except AttributeError:
            # Legacy policy interface (e.g. RandomActionPolicy): index only.
            action = actor.choose_action_index(env)
            logp, value = None, None
        if action is None:
            try:
                env.refresh()
            except Exception:
                break
            continue
        # Snapshot the candidate set the action was chosen from BEFORE stepping;
        # `env.step` rebuilds `available_actions`/`legal_action_mask` for the
        # next state, and the Q-head needs (obs_t, candidate-set_t) pairs.
        step_candidates = list(env.available_actions[:env.max_action_space])
        step_mask = np.asarray(env.legal_action_mask, dtype=np.int8).copy()
        try:
            _, reward, terminated, truncated, info = env.step(action)
        except Exception:
            metrics["ended_by"] = "step_error"
            break

        chosen = step_candidates[action] if 0 <= action < len(step_candidates) else None
        acting_seat = seat_key if collect else opp_seat_key
        prompt_now = (state_now.get("prompts") or {}).get(acting_seat) or {}
        _bump_action_metrics(metrics, chosen, prompt_now.get("menuTitle", ""))
        if chosen and str(chosen.get("actionType") or "") == "clickCard":
            hand = ((state_now.get("state") or {}).get(acting_seat) or {}).get("hand") or []
            card_uuid = chosen.get("uuid", "")
            if any(card.get("uuid") == card_uuid for card in hand):
                metrics["cards_played"] += 1
                if collect:
                    metrics["agent_cards_played"] += 1
        metrics["total_rewards"] += float(reward)
        metrics["total_reward_steps"] += 1
        if collect:
            metrics["agent_turns"] += 1
            metrics["agent_rewards"] += float(reward)
        else:
            metrics["opponent_turns"] += 1
            metrics["opponent_rewards"] += -float(reward)

        if collect and logp is not None:
            obs_list.append(obs_vec.numpy().astype(np.float32))
            actions.append(int(action))
            feature_list.append(np.asarray(
                [_action_features(entry) for entry in step_candidates],
                dtype=np.float32,
            ))
            mask_list.append(step_mask)
            rewards.append(float(reward))
        if truncated:
            metrics["ended_by"] = "truncated"
            break

    state = env.current_state or {}
    section = state.get("state") or {}
    agent_key = "player1" if str(state.get("player1Id")) == str(player_id) else "player2"
    opp_key = "player2" if agent_key == "player1" else "player1"

    agent_hp = _section_base_hp(section, agent_key)
    opp_hp = _section_base_hp(section, opp_key)

    # Prefer the server's winner list: `winners` holds Player.name
    # ('player1'/'player2') so compare it against the SEAT key, not the user id.
    # Endings that are not base destruction (deck-out, "has won" effects) can't
    # be classified from HP at all — those were the "unresolved" games.
    winners = [str(name) for name in (state.get("winners") or [])]
    if winners:
        winner = "agent" if agent_key in winners else "opponent"
    elif agent_hp <= 0 and opp_hp > 0:
        winner = "opponent"
    elif opp_hp <= 0 and agent_hp > 0:
        winner = "agent"
    elif agent_hp <= 0 and opp_hp <= 0:
        winner = "draw"
    else:
        winner = "unresolved"

    metrics["winner"] = winner
    metrics["agent_hp"] = agent_hp
    metrics["opp_hp"] = opp_hp
    metrics["winners"] = ",".join(winners)
    metrics["final_phase"] = state.get("phase")
    if metrics.get("ended_by") == "max_steps" and terminated:
        metrics["ended_by"] = "terminated"
    return (
        obs_list, actions, feature_list, mask_list, rewards, step, winner, agent_hp, opp_hp,
        p1_key, p2_key, metrics,
    )


def _parallel_worker(
    worker_id: int,
    server_url: str,
    player_id: str,
    obs_size: int,
    max_actions: int,
    log_dir: str,
    champion_path: str,
    self_play_probability: float,
    temperature: float,
    max_steps: int,
    decks_file: str,
    deck_keys: list,
    fixed_payload: dict,
    out_queue,
    stop_event,
):
    """Worker process: owns one env server, refreshes policy/champion weights
    from disk each episode, and pushes trajectories to the main process."""
    env = SWUEnv(server_url=server_url, player_id=player_id, single_agent_mode=True)
    policy = TorchPolicy(obs_size=obs_size, max_actions=max_actions, device="cpu", temperature=temperature)
    champion = TorchPolicy(obs_size=obs_size, max_actions=max_actions, device="cpu")
    weights_path = os.path.join(log_dir, "policy_worker.pt")
    last_weights_mtime = 0
    last_champion_mtime = 0

    while not stop_event.is_set():
        try:
            st = os.stat(weights_path)
            if st.st_mtime_ns != last_weights_mtime:
                policy.net.load_state_dict(_load_state_dict_any(weights_path))
                last_weights_mtime = st.st_mtime_ns
        except FileNotFoundError:
            pass
        try:
            st = os.stat(champion_path)
            if st.st_mtime_ns != last_champion_mtime:
                champion.net.load_state_dict(_load_state_dict_any(champion_path))
                last_champion_mtime = st.st_mtime_ns
        except FileNotFoundError:
            pass

        if random.random() < self_play_probability:
            opponent = _snapshot_policy(policy, "cpu")
            opponent_source = "self"
        else:
            opponent = champion
            opponent_source = "champion"

        if deck_keys:
            p1_key, p2_key = _sample_episode_decks(deck_keys)
            payload = _build_reset_payload(p1_key, p2_key, decks_file)[0]
        else:
            p1_key = p2_key = None
            payload = copy.deepcopy(fixed_payload)

        result = _run_worker_episode(env, policy, opponent, player_id, payload, max_steps,
                                     p1_key=p1_key, p2_key=p2_key, opponent_source=opponent_source)
        if result and result[0]:
            out_queue.put(result)


def _train_parallel(
    args,
    policy,
    champion_pool,
    env,
    logger,
    board,
    log_dir,
    champion_path,
    deck_keys,
    fixed_reset_payload,
    start_episode,
    verbose,
):
    """Async-A2C style: workers roll out episodes on their own env servers;
    the main process consumes trajectories, updates the policy, and runs the
    champion-gate tournaments on its own server (args.server_url)."""
    import multiprocessing as _mp

    base_port = _port_from_url(args.server_url)
    results: _mp.Queue = _mp.Queue(maxsize=args.num_workers * 8)
    stop = _mp.Event()
    workers = []
    for i in range(args.num_workers):
        worker = _mp.Process(
            target=_parallel_worker,
            args=(
                i,
                f"http://localhost:{base_port + 1 + i}",
                args.player_id,
                policy.obs_size,
                policy.max_actions,
                log_dir,
                champion_path,
                args.self_play_probability,
                args.temperature,
                args.max_steps,
                args.decks_file,
                deck_keys,
                fixed_reset_payload,
                results,
                stop,
            ),
            daemon=True,
        )
        worker.start()
        workers.append(worker)
    if verbose:
        print(f"[parallel] {args.num_workers} workers started (servers: {base_port + 1}..{base_port + args.num_workers}; "
              f"tournaments on {base_port})")

    weights_path = os.path.join(log_dir, "policy_worker.pt")
    _atomic_save_state_dict(weights_path, {key: value.detach().cpu() for key, value in policy.net.state_dict().items()})

    episodes_done = 0
    total_episodes = args.episodes
    last_episode_number = 0
    total_steps = 0
    batch_logps: list = []
    batch_returns: list = []
    batch_values: list = []
    last_losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
    last_tournament = None
    promotions = 0
    diagnostics_every = max(10, min(50, args.diagnostics_every))
    window_returns: list = []
    window_wins = 0.0
    window_episodes = 0
    deck_stats: dict[str, dict[str, Any]] = {}
    perf_t0 = time.time()
    perf_steps = 0
    perf_episodes = 0

    while episodes_done < total_episodes:
        try:
            result = results.get(timeout=1.0)
        except queue.Empty:
            continue
        obs_list, actions, feat_list, mask_list, rewards, steps, winner, agent_hp, opp_hp, p1_key, p2_key, episode_metrics = result
        episodes_done += 1
        last_episode_number = start_episode + episodes_done
        total_steps += steps
        perf_steps += steps
        perf_episodes += 1
        agent_reward = float(sum(rewards))

        if p1_key:
            st = deck_stats.setdefault(p1_key, {
                "games": 0, "wins": 0, "losses": 0, "draws": 0, "unresolved": 0, "opponents": {},
            })
            st["games"] += 1
            if winner == "agent":
                st["wins"] += 1
            elif winner == "opponent":
                st["losses"] += 1
            elif winner == "draw":
                st["draws"] += 1
            else:
                st["unresolved"] += 1
            if p2_key:
                matchup = st["opponents"].setdefault(p2_key, [0, 0])
                matchup[1] += 1
                if winner == "agent":
                    matchup[0] += 1

        if rewards:
            if winner == "unresolved":
                rewards[-1] += 0.5 * (agent_hp - opp_hp) / 30.0
            returns = discounted_returns(rewards, gamma=args.gamma)
            # Recompute Q-values under the CURRENT policy from the candidate
            # action-feature sets the worker chose among (async-A2C correction).
            obs_batch = torch.tensor(np.stack(obs_list), dtype=torch.float32)
            feat_tensors = [torch.tensor(feats, dtype=torch.float32) for feats in feat_list]
            q_logits, values = policy.evaluate_q(obs_batch, feat_tensors, mask_list)
            chosen = torch.tensor(actions, dtype=torch.long)
            logps = q_logits.gather(1, chosen.unsqueeze(1)).squeeze(1) - torch.logsumexp(q_logits, dim=1)
            batch_logps.extend(list(logps))
            batch_returns.extend(returns)
            batch_values.extend(list(values))

        window_returns.append(agent_reward)
        window_episodes += 1
        if winner == "agent":
            window_wins += 1.0

        board.scalar("episode/agent_return", agent_reward, last_episode_number)
        board.scalar("episode/steps", steps, last_episode_number)
        board.scalar("episode/win", 1.0 if winner == "agent" else 0.0, last_episode_number)
        # Write the SAME schema as the serial loop: the dashboards and notebooks
        # (visu.load_episode_summaries) read opponent_rewards / agent_turns /
        # ended_by from this CSV, and the header comes from the first record.
        opponent_reward_total = float(episode_metrics.get("opponent_rewards") or 0.0)
        agent_reward_total = float(episode_metrics.get("agent_rewards") or agent_reward)
        agent_turns = int(episode_metrics.get("agent_turns") or 0)
        opponent_turns = int(episode_metrics.get("opponent_turns") or 0)
        logger.record_episode_summary({
            **episode_metrics,
            "episode": last_episode_number,
            "winner": winner,
            "steps": steps,
            "agent_rewards": agent_reward_total,
            "opponent_rewards": opponent_reward_total,
            "agent_hp": agent_hp,
            "opp_hp": opp_hp,
            "p1_key": p1_key,
            "p2_key": p2_key,
            "agent_reward_per_turn": agent_reward_total / max(1, agent_turns),
            "opponent_reward_per_turn": opponent_reward_total / max(1, opponent_turns),
            "avg_valid_actions": int(episode_metrics.get("valid_actions_sum") or 0) / max(1, int(episode_metrics.get("valid_actions_count") or 0)),
            "avg_agent_valid_actions": int(episode_metrics.get("agent_valid_actions_sum") or 0) / max(1, int(episode_metrics.get("agent_valid_actions_count") or 0)),
            "cards_played_per_turn": int(episode_metrics.get("cards_played") or 0) / max(1, agent_turns + opponent_turns),
            "last_tournament_win_rate": last_tournament["candidate_win_rate"] if last_tournament else None,
            "promotions": promotions,
        })

        # ── A2C update ──
        if len(batch_logps) > 0 and (episodes_done % args.update_every == 0 or episodes_done >= total_episodes):
            total_loss, policy_loss, value_loss, entropy = policy.update(
                batch_logps, batch_returns, batch_values,
                value_coef=args.value_coef, entropy_coef=args.entropy_coef,
            )
            last_losses = {"policy": policy_loss, "value": value_loss, "entropy": entropy, "total": total_loss}
            n = len(batch_logps)
            if verbose:
                print(f"Batch update after episode {last_episode_number} ({n} steps): "
                      f"total={total_loss:.4f} policy={policy_loss:.4f} value={value_loss:.4f} entropy={entropy:.4f}")
            for tag, value in (("train/total_loss", total_loss), ("train/policy_loss", policy_loss),
                               ("train/value_loss", value_loss), ("train/entropy", entropy)):
                board.scalar(tag, value, last_episode_number)
            batch_logps, batch_returns, batch_values = [], [], []
            _atomic_save_state_dict(weights_path, {key: value.detach().cpu() for key, value in policy.net.state_dict().items()})
            latest_payload = {
                "model_state_dict": {key: value.detach().cpu() for key, value in policy.net.state_dict().items()},
                "optimizer_state_dict": policy.optimizer.state_dict(),
                "episode": last_episode_number,
                "obs_size": policy.obs_size,
                "max_actions": policy.max_actions,
                "checkpoint_source": args.checkpoint,
            }
            torch.save(latest_payload, os.path.join(log_dir, "policy_latest.ckpt"))

        # ── Evaluation tournament & champion gate ──
        if args.tournament_every > 0 and (last_episode_number % args.tournament_every == 0 or episodes_done >= total_episodes):
            if verbose:
                print(f"[tournament] starting {args.tournament_games}-game evaluation (candidate vs champion) after episode {last_episode_number}")
            if deck_keys:
                make_reset_payload = lambda: _build_reset_payload(*_sample_episode_decks(deck_keys), args.decks_file)[0]
            else:
                make_reset_payload = lambda: copy.deepcopy(fixed_reset_payload)
            last_tournament = run_tournament(
                env,
                candidate=policy,
                champion=champion_pool.champion,
                reset_payload_factory=make_reset_payload,
                games=args.tournament_games,
                max_steps=args.max_steps,
                stall_polls=args.stall_polls,
                verbose=verbose,
            )
            win_rate = last_tournament["candidate_win_rate"]
            if verbose:
                print(f"[tournament] result: candidate {last_tournament['candidate_wins']} wins, "
                      f"champion {last_tournament['champion_wins']} wins, draws {last_tournament['draws']}, "
                      f"unresolved {last_tournament['unresolved']} — candidate win rate {win_rate:.1%}")
            board.scalar("eval/candidate_win_rate", win_rate, last_episode_number)
            if win_rate >= args.promote_win_rate:
                champion_pool.save_champion(
                    policy,
                    episode=last_episode_number,
                    reason=f"promoted: win rate {win_rate:.1%} >= {args.promote_win_rate:.1%}",
                )
                promotions += 1
                if verbose:
                    print(f"[promotion] candidate replaced the champion at episode {last_episode_number}")

        # ── Diagnostics window ──
        if episodes_done % diagnostics_every == 0 or episodes_done >= total_episodes:
            avg_return = sum(window_returns) / max(1, window_episodes)
            win_rate_pct = 100.0 * window_wins / max(1, window_episodes)
            champion_wins_display = str(last_tournament["champion_wins"]) if last_tournament is not None else "-"
            print(f"[Episode {last_episode_number}] Avg Return: {avg_return:+.3f} | Win Rate: {win_rate_pct:.1f}% | "
                  f"Policy Loss: {last_losses['policy']:.4f} | Value Loss: {last_losses['value']:.4f} | "
                  f"Champion Wins: {champion_wins_display}")
            window_returns, window_wins, window_episodes = [], 0.0, 0

        # ── Perf report ──
        if args.perf_every > 0 and perf_episodes >= args.perf_every:
            _perf_line(perf_t0, perf_steps, perf_episodes, last_episode_number)
            perf_t0, perf_steps, perf_episodes = time.time(), 0, 0

    stop.set()
    for worker in workers:
        worker.join(timeout=10)
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    _write_deck_report(log_dir, deck_stats, episodes_done)
    if verbose:
        print(f"Training finished (parallel, {args.num_workers} workers, {total_steps} steps). "
              f"Champion file: {champion_path} (promotions: {promotions})")


# ── Deck meta report ─────────────────────────────────────────────────────────
def _decisive(deck: dict[str, Any]) -> int:
    return max(0, int(deck["games"]) - int(deck["unresolved"]))


def _write_deck_report(log_dir: str, deck_stats: dict[str, dict[str, Any]], episodes: int = 0) -> None:
    """Print + save a small deck meta report: winrate-ranked decklist and a
    matchup matrix (agent deck rows vs opponent deck cols)."""
    if not deck_stats:
        print("\n[deck report] no randomized-deck episodes recorded — skipping")
        return

    def _winrate(deck: dict[str, Any]) -> float:
        decisive = _decisive(deck)
        return deck["wins"] / decisive if decisive > 0 else 0.0

    ordered = sorted(
        deck_stats.items(),
        key=lambda kv: (_winrate(kv[1]), kv[1]["games"]),
        reverse=True,
    )
    decks = [key for key, _ in ordered]

    lines: list[str] = []
    lines.append("DECK WINRATE REPORT (agent side; decisive = games - unresolved)")
    lines.append("=" * 72)
    lines.append(f"{'deck':<40}{'games':>7}{'wins':>6}{'losses':>8}{'draws':>7}{'unres':>7}{'win%':>8}")
    for key, st in ordered:
        lines.append(
            f"{str(key)[:40]:<40}{st['games']:>7}{st['wins']:>6}{st['losses']:>8}"
            f"{st['draws']:>7}{st['unresolved']:>7}{_winrate(st):>8.1%}"
        )

    # ── Matchup matrix (only decks that appeared at least once) ──
    lines.append("")
    lines.append("MATCHUP MATRIX — agent deck (row) vs opponent deck (col), cell = agent win% (games in parens)")
    lines.append("=" * 72)
    width = 22
    header_cells = [str(key)[:width].ljust(width) for key in decks]
    lines.append(" " * width + "".join(header_cells))
    for row_key in decks:
        st = deck_stats[row_key]
        row_cells = []
        for col_key in decks:
            wins, games = st["opponents"].get(col_key, (0, 0))
            if games > 0:
                cell = f"{wins / games:.0%}({games})"[:width - 1].rjust(width - 1)
            else:
                cell = "-".rjust(width - 1)
            row_cells.append(cell)
        lines.append(str(row_key)[:width].ljust(width) + "".join(row_cells))

    report_path = os.path.join(log_dir, "deck_report.txt")
    try:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except OSError:
        pass

    # ── Console summary ──
    print("\n[deck report] (full report saved to deck_report.txt)")
    shown = min(12, len(ordered))
    for key, st in ordered[:shown]:
        print(f"  {str(key)[:40]:<40} {_winrate(st):>6.1%}  ({st['wins']}W/{st['losses']}L/{st['draws']}D, "
              f"{st['games']} games)")
    if len(ordered) > shown:
        worst = ordered[-3:]
        print(f"  ... ({len(ordered) - shown - len(worst)} more)")
        for key, st in reversed(worst):
            print(f"  {str(key)[:40]:<40} {_winrate(st):>6.1%}  ({st['wins']}W/{st['losses']}L/{st['draws']}D, "
                  f"{st['games']} games)")

    # ── CSV for pivoting ──
    csv_path = os.path.join(log_dir, "deck_matchups.csv")
    try:
        with open(csv_path, "w", encoding="utf-8") as handle:
            handle.write("agent_deck,opponent_deck,games,agent_wins,winrate\n")
            for key, st in ordered:
                for opp_key, (wins, games) in sorted(st["opponents"].items(), key=lambda kv: -kv[1][1]):
                    if games > 0:
                        handle.write(f"{key},{opp_key},{games},{wins},{wins / games:.3f}\n")
    except OSError:
        pass

    # ── Heatmap PNG (optional; skipped when matplotlib is missing or too many decks) ──
    if len(decks) < 2:
        return
    if len(decks) > 90:
        print(f"[deck report] matrix too large for a readable heatmap ({len(decks)} decks) — text + CSV only")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    try:
        import numpy as _np
        matrix = _np.full((len(decks), len(decks)), _np.nan)
        for row_idx, row_key in enumerate(decks):
            st = deck_stats[row_key]
            for col_idx, col_key in enumerate(decks):
                wins, games = st["opponents"].get(col_key, (0, 0))
                if games > 0:
                    matrix[row_idx, col_idx] = wins / games
        size = max(8.0, len(decks) * 0.35)
        fig, ax = plt.subplots(figsize=(size, size * 0.92))
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#dddddd")
        image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(decks)))
        ax.set_yticks(range(len(decks)))
        ax.set_xticklabels(decks, rotation=90, fontsize=max(4.0, 140.0 / size))
        ax.set_yticklabels(decks, fontsize=max(4.0, 140.0 / size))
        if len(decks) <= 24:
            for row_idx in range(len(decks)):
                for col_idx in range(len(decks)):
                    value = matrix[row_idx, col_idx]
                    if not _np.isnan(value):
                        ax.text(col_idx, row_idx, f"{value:.0%}",
                                ha="center", va="center", fontsize=max(4.0, 110.0 / size))
        ax.set_title(f"Deck matchup win-rate (agent rows vs opponent cols) — {episodes} episodes")
        fig.colorbar(image, ax=ax, shrink=0.8, label="agent win rate")
        fig.tight_layout()
        heatmap_path = os.path.join(log_dir, "deck_heatmap.png")
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        print(f"[deck report] heatmap saved to deck_heatmap.png ({len(decks)}x{len(decks)})")
    except Exception as exc:  # never let a report crash training
        print(f"[deck report] heatmap rendering failed: {exc}")


# ── Main training loop ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", default="http://localhost:3005")
    parser.add_argument("--player_id", default="111")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_dir", default="runs/train")
    parser.add_argument("--decks_file", type=str, default="decks.json")
    parser.add_argument("--p1", type=str, help="Deck key for player 1")
    parser.add_argument("--p2", type=str, help="Deck key for player 2")
    parser.add_argument("--randomize_decks", action="store_true", help="Sample fresh decks from decks.json for every episode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from a saved checkpoint (.pt or .ckpt)")
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Archive candidate weights into the checkpoints history every N episodes")
    parser.add_argument("--stall_polls", type=int, default=40, help="Abort an episode after this many repeated no-action polls in the same prompt state")
    parser.add_argument("--update_every", type=int, default=1, help="Accumulate this many episodes before each policy update (minibatch). 1 = per-episode update, 8-16 recommended for stability")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-episode chatter; keep diagnostics summary lines")
    parser.add_argument("--debug_steps", action="store_true", help="Re-enable verbose per-step action dumps (default: suppressed)")

    # Gated champion pool
    parser.add_argument("--champion_path", type=str, default="", help=f"Champion checkpoint path (default: <log_dir>/{CHAMPION_FILENAME})")
    parser.add_argument("--champion_probability", type=float, default=0.8, help="Fraction of training episodes against the champion (rest: random history checkpoint)")
    parser.add_argument("--self_play_probability", type=float, default=0.5, help="Fraction of episodes where the opponent is a snapshot of the candidate itself (self-play); keeps games real so win/loss signals actually occur")
    parser.add_argument("--tournament_every", type=int, default=500, help="Run the evaluation tournament every N episodes (also after the final episode)")
    parser.add_argument("--tournament_games", type=int, default=50, help="Games per evaluation tournament (P1/P2 seats alternate evenly)")
    parser.add_argument("--promote_win_rate", type=float, default=0.55, help="Candidate win rate threshold for champion promotion")

    # A2C loss coefficients
    parser.add_argument("--value_coef", type=float, default=0.5, help="c1: critic loss weight")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="c2: entropy bonus weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling softmax temperature (1.0 = unchanged; >1 stops the policy from saturating into a deterministic loop)")

    # Diagnostics
    parser.add_argument("--diagnostics_every", type=int, default=20, help="Print the clean summary line every N episodes (clamped to 10-50)")
    parser.add_argument("--no_tensorboard", action="store_true", help="Disable TensorBoard logging even when available")
    parser.add_argument("--num_workers", type=int, default=1, help="Parallel game workers, each with its own env server on port server_port+1..+N (tournaments use server_port). 1 = serial mode. Start N+1 servers.")
    parser.add_argument("--perf_every", type=int, default=50, help="Print system perf (eps/s, steps/s, RAM, server processes, GPU) every N episodes (0 = off)")

    args = parser.parse_args()

    # The RL env server falls back to EMPTY decks when no cards are supplied,
    # which leaves the game stuck in the setup resource step forever.
    if not args.randomize_decks and (not args.p1 or not args.p2):
        parser.error(
            "No decks configured: provide both --p1 <deck_key> and --p2 <deck_key>, "
            "or use --randomize_decks (the RL server's empty-deck default cannot "
            "complete the setup phase)."
        )

    verbose = not args.quiet
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    champion_path = args.champion_path or os.path.join(log_dir, CHAMPION_FILENAME)

    logger = EpisodeLogger(log_dir=log_dir, verbose=verbose)
    env = SWUEnv(server_url=args.server_url, player_id=args.player_id, single_agent_mode=True)
    policy = TorchPolicy(
        obs_size=env.observation_space.shape[0],
        max_actions=env.action_space.n,
        lr=args.lr,
        device=args.device,
        temperature=args.temperature,
    )

    champion_pool = ChampionPool(
        log_dir=log_dir,
        champion_path=champion_path,
        obs_size=policy.obs_size,
        max_actions=policy.max_actions,
        device=args.device,
        champion_probability=args.champion_probability,
        self_play_probability=args.self_play_probability,
        verbose=verbose,
    )

    board = MetricsBoard(TENSORBOARD_AVAILABLE and not args.no_tensorboard, log_dir)

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

    # Seed the champion from the current candidate when no champion exists yet.
    if not champion_pool.has_champion:
        champion_pool.save_champion(policy, episode=start_episode, reason="initial champion (no champion file found)")

    deck_keys = _load_deck_keys(args.decks_file) if args.randomize_decks else []
    fixed_reset_payload, fixed_deck_meta = _build_reset_payload(args.p1, args.p2, args.decks_file)

    def make_reset_payload() -> dict:
        if args.randomize_decks:
            p1_key, p2_key = _sample_episode_decks(deck_keys)
            return _build_reset_payload(p1_key, p2_key, args.decks_file)[0]
        return copy.deepcopy(fixed_reset_payload)

    # Batch accumulators for the A2C update
    batch_logps: list[torch.Tensor] = []
    batch_returns: list[torch.Tensor] = []
    batch_values: list[torch.Tensor] = []

    # Diagnostics window + last-known loss components
    diagnostics_every = max(10, min(50, args.diagnostics_every))
    window_returns: list[float] = []
    window_wins = 0.0
    window_episodes = 0
    last_losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "total": 0.0}
    last_tournament: dict[str, Any] | None = None
    promotions = 0
    perf_t0 = time.time()
    perf_steps = 0

    # ── Parallel mode: workers roll out games on their own env servers ──
    if args.num_workers > 1:
        _train_parallel(
            args,
            policy=policy,
            champion_pool=champion_pool,
            env=env,
            logger=logger,
            board=board,
            log_dir=log_dir,
            champion_path=champion_path,
            deck_keys=deck_keys,
            fixed_reset_payload=fixed_reset_payload,
            start_episode=start_episode,
            verbose=verbose,
        )
        board.close()
        logger.close()
        return

    for ep in range(args.episodes):
        episode_number = start_episode + ep + 1
        reset_payload = make_reset_payload()
        obs, info = env.reset(options=reset_payload)
        if verbose:
            print(f"=== Episode {episode_number}/{start_episode + args.episodes} ===")
        logger.record_rl_transition({
            "event": "reset",
            "player_id": args.player_id,
            "state": env.current_state,
            "available_actions": _slim_actions(env.available_actions),
            "info": _slim_info(info),
            "decks": fixed_deck_meta if not args.randomize_decks else {},
        })

        # ── Opponent sampling: self-play snapshot / champion / history ──
        opponent_policy, opponent_source = champion_pool.sample_opponent(candidate=policy)
        opponent = PolicyOpponent(opponent_policy)
        if verbose:
            print(f"[opponent] episode {episode_number} opponent source: {opponent_source}")

        logps: list[torch.Tensor] = []
        rewards: list[float] = []
        values: list[torch.Tensor] = []
        step_idx = 0
        terminated = False
        step_info = None

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
            "agent_hand_sum": 0.0,
            "opp_hand_sum": 0.0,
            "agent_max_valid_actions": 0,
            "final_phase": None,
            "winner": None,
            "cards_played": 0,
            "agent_cards_played": 0,
            "cancel_clicks": 0,
            "pass_clicks": 0,
            "done_clicks": 0,
            "attack_clicks": 0,
        }

        no_action_poll_count = 0
        last_stall_signature = None

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
            elif valid_actions == 0 and active is None:
                agent_key = _player_key_for_id(state_snapshot, args.player_id)
                agent_prompt = (prompt_snapshot.get(agent_key) if agent_key else None) or {}
                if agent_prompt and "waiting for opponent" not in str(agent_prompt.get("menuTitle", "")).lower():
                    if stall_signature == last_stall_signature:
                        no_action_poll_count += 1
                    else:
                        no_action_poll_count = 1
                    last_stall_signature = stall_signature
                else:
                    no_action_poll_count = 0
                    last_stall_signature = None
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
            episode_metrics["agent_hand_sum"] += agent_board["hand_count"]
            episode_metrics["opp_base_hp_sum"] += opp_board["base_hp"]
            episode_metrics["opp_leader_hp_sum"] += opp_board["leader_hp"]
            episode_metrics["opp_board_power_sum"] += opp_board["board_power"]
            episode_metrics["opp_board_hp_sum"] += opp_board["board_hp"]
            episode_metrics["opp_board_damage_sum"] += opp_board["board_damage"]
            episode_metrics["opp_unit_count_sum"] += opp_board["unit_count"]
            episode_metrics["opp_exhausted_sum"] += opp_board["exhausted_count"]
            episode_metrics["opp_hand_sum"] += opp_board["hand_count"]

            if args.debug_steps:
                logger.log(
                    f"[loop] step={step_idx} phase={phase} activePlayer={active} valid_actions={len(env.available_actions)} "
                    f"prompt1={str((prompt_snapshot.get('player1') or {}).get('menuTitle', ''))!r} "
                    f"prompt2={str((prompt_snapshot.get('player2') or {}).get('menuTitle', ''))!r}",
                    player_id=args.player_id,
                )
                _log_available_actions(logger, args.player_id, list(env.available_actions))

            if no_action_poll_count >= args.stall_polls and str(active) == str(args.player_id):
                logger.log(
                    f"[warning] Stalled prompt detected after {no_action_poll_count} polls with zero legal actions. Aborting episode.",
                    player_id=args.player_id,
                )
                logger.record_rl_transition({
                    "event": "episode_abort",
                    "reason": "stalled_no_action",
                    "player_id": args.player_id,
                    "step_index": step_idx,
                    "state": state_snapshot,
                    "available_actions": _slim_actions(env.available_actions),
                    "info": _slim_info(info),
                    "stall_polls": no_action_poll_count,
                })
                episode_metrics["final_phase"] = phase
                terminated = True
                break

            if str(active) == str(args.player_id):
                # ── Agent's turn (candidate) ──
                episode_metrics["agent_turns"] += 1
                episode_metrics["agent_valid_actions_sum"] += valid_actions
                episode_metrics["agent_valid_actions_count"] += 1
                obs_vec = torch.tensor(env._get_obs(), dtype=torch.float32)
                available_actions = list(env.available_actions)
                action, logp, value = policy.select_action(
                    obs_vec, available_actions, getattr(env, "legal_action_mask", None)
                )
                if action is None:
                    if no_action_poll_count >= args.stall_polls:
                        logger.log(f"[warning] Agent stall aborted episode at step {step_idx}", player_id=args.player_id)
                        episode_metrics["final_phase"] = phase
                        terminated = True
                        break
                    try:
                        env.refresh()
                    except Exception as exc:
                        logger.log(f"[agent] refresh failed while waiting for actions: {exc}", player_id=args.player_id)
                        episode_metrics["final_phase"] = phase
                        terminated = True
                        break
                    # time.sleep(0.01)
                    continue

                if action >= len(available_actions) or action < 0:
                    logger.log(f"Policy produced invalid action {action} for {len(available_actions)} available", player_id=args.player_id)
                    # time.sleep(0.01)
                    continue

                chosen_action = available_actions[action]
                _bump_action_metrics(
                    episode_metrics,
                    chosen_action,
                    (prompt_snapshot.get(agent_key) or {}).get("menuTitle", "") if agent_key else "",
                )
                if args.debug_steps:
                    logger.log(f"[agent] p1 chose [{action}] {_describe_action(chosen_action, action)}", player_id=args.player_id)

                try:
                    _, reward, terminated, truncated, step_info = env.step(action)
                except Exception as exc:
                    logger.log(f"[agent] step failed; skipping episode: {exc}", player_id=args.player_id)
                    logger.record_rl_transition({
                        "event": "step_error",
                        "player_id": args.player_id,
                        "step_index": step_idx,
                        "state": state_snapshot,
                        "available_actions": _slim_actions(available_actions),
                        "action_index": action,
                        "action": chosen_action,
                        "reward": -10.0,
                        "terminated": True,
                        "truncated": False,
                        "next_state": copy.deepcopy(env.current_state),
                        "info": _slim_info(info),
                        "error": str(exc),
                    })
                    episode_metrics["final_phase"] = phase
                    episode_metrics["agent_rewards"] += -10.0
                    episode_metrics["total_rewards"] += -10.0
                    terminated = True
                    step_info = info
                    break

                episode_metrics["total_rewards"] += float(reward)
                episode_metrics["total_reward_steps"] += 1
                episode_metrics["agent_rewards"] += float(reward)

                action_type = str(chosen_action.get("actionType") or "")
                if action_type == "clickCard":
                    card_uuid = chosen_action.get("uuid", "")
                    agent_state = (state_section.get(agent_key) if agent_key else None) or {}
                    in_hand = any(c.get("uuid") == card_uuid for c in agent_state.get("hand", []))
                    if in_hand:
                        episode_metrics["cards_played"] += 1
                        episode_metrics["agent_cards_played"] += 1

                state_section = (step_info or {}).get("state_dict") or env.current_state or {}
                if (step_info or {}).get("no_progress"):
                    _log_no_progress(logger, args.player_id, step_idx, chosen_action, env)
                logger.record_rl_transition({
                    "event": "step",
                    "player_id": args.player_id,
                    "step_index": step_idx,
                    "state": state_snapshot,
                    "available_actions": _slim_actions(available_actions),
                    "action_index": action,
                    "action": chosen_action,
                    "reward": reward,
                    "terminated": terminated,
                    "info": _slim_info(step_info),
                })
                logger.record_step_analysis_data({
                    "episode": episode_number,
                    "step_index": step_idx,
                    "active_player_id": str(active),
                    "acting_player_id": args.player_id,
                    "phase": phase,
                    "valid_actions_count": valid_actions,
                    "agent_base_hp": agent_board["base_hp"],
                    "agent_board_power": agent_board["board_power"],
                    "agent_board_hp": agent_board["board_hp"],
                    "agent_unit_count": agent_board["unit_count"],
                    "agent_ready_resources": agent_board["ready_resources"],
                    "agent_credits": agent_board["credits"],
                    "agent_hand_count": agent_board["hand_count"],
                    "opp_base_hp": opp_board["base_hp"],
                    "opp_board_power": opp_board["board_power"],
                    "opp_board_hp": opp_board["board_hp"],
                    "opp_unit_count": opp_board["unit_count"],
                    "reward": float(reward),
                    "terminated": terminated,
                    "truncated": truncated,
                })

                # Collect (log_prob, value, reward) for the A2C update.
                if logp is not None:
                    logps.append(logp)
                    values.append(value)
                    rewards.append(float(reward))
            else:
                # ── Opponent's turn (champion / history checkpoint) ──
                episode_metrics["opponent_turns"] += 1
                action = opponent.choose_action_index(env)
                if action is None:
                    if no_action_poll_count >= args.stall_polls:
                        logger.log(f"[warning] Opponent stall aborted episode at step {step_idx}", player_id=args.player_id)
                        episode_metrics["final_phase"] = phase
                        terminated = True
                        break
                    try:
                        env.refresh()
                    except Exception as exc:
                        logger.log(f"[opponent] refresh failed while waiting for actions: {exc}", player_id=args.player_id)
                        episode_metrics["final_phase"] = phase
                        terminated = True
                        break
                    # time.sleep(0.01)
                    continue

                opponent_action = list(env.available_actions)[action] if 0 <= action < len(env.available_actions) else None
                _bump_action_metrics(
                    episode_metrics,
                    opponent_action,
                    (prompt_snapshot.get(opp_key) or {}).get("menuTitle", "") if opp_key else "",
                )
                if args.debug_steps and opponent_action is not None:
                    logger.log(f"[opponent] p2 chose [{action}] {_describe_action(opponent_action, action)}", player_id=args.player_id)

                try:
                    _, reward, terminated, truncated, step_info = env.step(action)
                except Exception as exc:
                    logger.log(f"[opponent] step failed; aborting episode: {exc}", player_id=args.player_id)
                    logger.record_rl_transition({
                        "event": "step_error",
                        "player_id": "opponent",
                        "step_index": step_idx,
                        "state": state_snapshot,
                        "available_actions": _slim_actions(env.available_actions),
                        "action_index": action,
                        "action": opponent_action,
                        "reward": -10.0,
                        "terminated": True,
                        "truncated": False,
                        "next_state": copy.deepcopy(env.current_state),
                        "info": _slim_info(info),
                        "error": str(exc),
                    })
                    episode_metrics["final_phase"] = phase
                    episode_metrics["opponent_rewards"] += 10.0
                    episode_metrics["total_rewards"] += -10.0
                    terminated = True
                    step_info = info
                    break

                episode_metrics["total_rewards"] += float(reward)
                episode_metrics["total_reward_steps"] += 1
                # `reward` is shaped from the agent's perspective; negate it for
                # the opponent-perspective bookkeeping column.
                episode_metrics["opponent_rewards"] += -float(reward)

                if opponent_action and str(opponent_action.get("actionType") or "") == "clickCard":
                    card_uuid = opponent_action.get("uuid", "")
                    opp_state = (state_section.get(opp_key) if opp_key else None) or {}
                    in_hand = any(c.get("uuid") == card_uuid for c in opp_state.get("hand", []))
                    if in_hand:
                        episode_metrics["cards_played"] += 1

                if (step_info or {}).get("no_progress"):
                    _log_no_progress(logger, "opponent", step_idx, opponent_action or {}, env)

                logger.record_rl_transition({
                    "event": "step",
                    "player_id": "opponent",
                    "step_index": step_idx,
                    "state": state_snapshot,
                    "available_actions": _slim_actions(env.available_actions),
                    "action_index": action,
                    "action": opponent_action,
                    "reward": reward,
                    "terminated": terminated,
                    "info": _slim_info(step_info),
                })
                logger.record_step_analysis_data({
                    "episode": episode_number,
                    "step_index": step_idx,
                    "active_player_id": str(active),
                    "acting_player_id": "opponent",
                    "phase": phase,
                    "valid_actions_count": valid_actions,
                    "agent_base_hp": agent_board["base_hp"],
                    "agent_board_power": agent_board["board_power"],
                    "agent_board_hp": agent_board["board_hp"],
                    "agent_unit_count": agent_board["unit_count"],
                    "agent_ready_resources": agent_board["ready_resources"],
                    "agent_credits": agent_board["credits"],
                    "agent_hand_count": agent_board["hand_count"],
                    "opp_base_hp": opp_board["base_hp"],
                    "opp_board_power": opp_board["board_power"],
                    "opp_board_hp": opp_board["board_hp"],
                    "opp_unit_count": opp_board["unit_count"],
                    "reward": float(reward),
                    "terminated": terminated,
                    "truncated": truncated,
                })

            step_idx += 1

            if terminated or truncated:
                episode_metrics["final_phase"] = (step_info or {}).get("phase") if isinstance(step_info, dict) else None
                winners = (env.current_state or {}).get("winners", [])
                if winners:
                    episode_metrics["winner"] = winners[0] if len(winners) == 1 else winners

        # ── Episode end: winner resolution ──
        final_state = (env.current_state or {}).get("state") or {}
        final_agent = _unit_board_metrics(final_state, _player_key_for_id(env.current_state, args.player_id) or "player1")
        agent_key = _player_key_for_id(env.current_state, args.player_id)
        final_opp = _unit_board_metrics(final_state, "player2" if agent_key == "player1" else "player1")
        agent_base_dead = final_agent["base_hp"] <= 0.0
        opp_base_dead = final_opp["base_hp"] <= 0.0
        if agent_base_dead and not opp_base_dead:
            episode_metrics["winner"] = "opponent"
        elif opp_base_dead and not agent_base_dead:
            episode_metrics["winner"] = "agent"
        elif agent_base_dead and opp_base_dead:
            episode_metrics["winner"] = "draw"
        else:
            episode_metrics["winner"] = "unresolved"

        # Accumulate the episode into the A2C batch.
        if len(rewards) > 0:
            if episode_metrics["winner"] == "unresolved":
                # The episode was truncated at max_steps, so the env never
                # produced its ±10 win/loss signal. Score the final board so the
                # policy still learns that damaging the enemy base is good:
                # leading on base HP → positive, trailing → negative.
                bonus = 0.5 * (final_agent["base_hp"] - final_opp["base_hp"]) / 30.0
                rewards[-1] += bonus
            returns = discounted_returns(rewards, gamma=args.gamma)
            batch_logps.extend(logps)
            batch_returns.extend(returns)
            batch_values.extend(values)

        # Archive candidate weights into the history folder (for future sampling).
        if args.checkpoint_every > 0 and episode_number % args.checkpoint_every == 0:
            champion_pool.register_checkpoint(policy, episode_number)

        # ── A2C update ──
        do_update = (
            len(batch_logps) > 0
            and (
                (ep + 1) % args.update_every == 0
                or ep == args.episodes - 1
            )
        )
        if do_update:
            total_loss, policy_loss, value_loss, entropy = policy.update(
                batch_logps,
                batch_returns,
                batch_values,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
            )
            last_losses = {"policy": policy_loss, "value": value_loss, "entropy": entropy, "total": total_loss}
            n = len(batch_logps)
            if verbose:
                print(f"Batch update after episode {episode_number} ({n} steps): "
                      f"total={total_loss:.4f} policy={policy_loss:.4f} value={value_loss:.4f} entropy={entropy:.4f}")
            board.scalar("train/total_loss", total_loss, episode_number)
            board.scalar("train/policy_loss", policy_loss, episode_number)
            board.scalar("train/value_loss", value_loss, episode_number)
            board.scalar("train/entropy", entropy, episode_number)
            batch_logps = []
            batch_returns = []
            batch_values = []

        # ── Evaluation tournament & champion gate ──
        if args.tournament_every > 0 and (episode_number % args.tournament_every == 0 or ep == args.episodes - 1):
            if verbose:
                print(f"[tournament] starting {args.tournament_games}-game evaluation (candidate vs champion) after episode {episode_number}")
            last_tournament = run_tournament(
                env,
                candidate=policy,
                champion=champion_pool.champion,
                reset_payload_factory=make_reset_payload,
                games=args.tournament_games,
                max_steps=args.max_steps,
                stall_polls=args.stall_polls,
                verbose=verbose,
            )
            win_rate = last_tournament["candidate_win_rate"]
            if verbose:
                print(f"[tournament] result: candidate {last_tournament['candidate_wins']} wins, "
                      f"champion {last_tournament['champion_wins']} wins, draws {last_tournament['draws']}, "
                      f"unresolved {last_tournament['unresolved']} — candidate win rate {win_rate:.1%}")
            board.scalar("eval/candidate_win_rate", win_rate, episode_number)
            board.scalar("eval/champion_wins", float(last_tournament["champion_wins"]), episode_number)

            if win_rate > args.promote_win_rate:
                champion_pool.save_champion(
                    policy,
                    episode=episode_number,
                    reason=f"promoted: win rate {win_rate:.1%} > {args.promote_win_rate:.1%}",
                )
                promotions += 1
                board.scalar("eval/promotion", 1.0, episode_number)
                if verbose:
                    print(f"[promotion] candidate replaced the champion at episode {episode_number}")
            else:
                board.scalar("eval/promotion", 0.0, episode_number)

        # ── Diagnostics window ──
        window_returns.append(float(episode_metrics["agent_rewards"]))
        window_episodes += 1
        if episode_metrics["winner"] == "agent":
            window_wins += 1.0

        board.scalar("episode/agent_return", float(episode_metrics["agent_rewards"]), episode_number)
        board.scalar("episode/steps", step_idx, episode_number)
        board.scalar("episode/win", 1.0 if episode_metrics["winner"] == "agent" else 0.0, episode_number)
        board.scalar(
            "episode/avg_valid_actions",
            episode_metrics["valid_actions_sum"] / max(1, episode_metrics["valid_actions_count"]),
            episode_number,
        )

        summary = {
            **episode_metrics,
            "steps": step_idx,
            "agent_reward_per_turn": episode_metrics["agent_rewards"] / max(1, episode_metrics["agent_turns"]),
            "opponent_reward_per_turn": episode_metrics["opponent_rewards"] / max(1, episode_metrics["opponent_turns"]),
            "avg_valid_actions": episode_metrics["valid_actions_sum"] / max(1, episode_metrics["valid_actions_count"]),
            "avg_agent_valid_actions": episode_metrics["agent_valid_actions_sum"] / max(1, episode_metrics["agent_valid_actions_count"]),
            "cards_played_per_turn": episode_metrics["cards_played"] / max(1, episode_metrics["agent_turns"] + episode_metrics["opponent_turns"]),
            "last_tournament_win_rate": last_tournament["candidate_win_rate"] if last_tournament else None,
            "promotions": promotions,
        }
        logger.record_episode_summary(summary)

        if verbose:
            maxed = step_idx >= args.max_steps
            print(
                f"[episode {episode_number}] steps={step_idx}{' (max_steps)' if maxed else ''} "
                f"winner={episode_metrics['winner']} "
                f"| bases: agent {final_agent['base_hp']:.0f} HP ({final_agent['board_damage']:.0f} dmg dealt), "
                f"opp {final_opp['base_hp']:.0f} HP ({final_opp['board_damage']:.0f} dmg dealt) "
                f"| agent rew {episode_metrics['agent_rewards']:+.2f} ({episode_metrics['agent_turns']} turns), "
                f"opp rew {episode_metrics['opponent_rewards']:+.2f} ({episode_metrics['opponent_turns']} turns) "
                f"| cancel={episode_metrics['cancel_clicks']} pass={episode_metrics['pass_clicks']} "
                f"done={episode_metrics['done_clicks']} attack={episode_metrics['attack_clicks']}"
            )

        if episode_number % diagnostics_every == 0 or ep == args.episodes - 1:
            avg_return = sum(window_returns) / max(1, window_episodes)
            win_rate_pct = 100.0 * window_wins / max(1, window_episodes)
            champion_wins_display = str(last_tournament["champion_wins"]) if last_tournament is not None else "-"
            print(
                f"[Episode {episode_number}] Avg Return: {avg_return:+.3f} | Win Rate: {win_rate_pct:.1f}% | "
                f"Policy Loss: {last_losses['policy']:.4f} | Value Loss: {last_losses['value']:.4f} | "
                f"Champion Wins: {champion_wins_display}"
            )
            window_returns = []
            window_wins = 0.0
            window_episodes = 0

        # ── Perf report (serial mode) ──
        perf_steps += step_idx
        if args.perf_every > 0 and episode_number % args.perf_every == 0:
            perf_t0, perf_steps = _perf_line(perf_t0, perf_steps, args.perf_every, episode_number)

        # ── Persist candidate checkpoints ──
        latest_payload = {
            "model_state_dict": policy.net.state_dict(),
            "optimizer_state_dict": policy.optimizer.state_dict(),
            "episode": episode_number,
            "obs_size": policy.obs_size,
            "max_actions": policy.max_actions,
            "checkpoint_source": args.checkpoint,
        }
        torch.save(policy.net.state_dict(), os.path.join(log_dir, "policy_latest.pt"))
        torch.save(latest_payload, os.path.join(log_dir, "policy_latest.ckpt"))

    board.close()
    logger.close()
    if verbose:
        print(f"Training finished. Champion file: {champion_path} (promotions: {promotions})")


if __name__ == "__main__":
    main()

