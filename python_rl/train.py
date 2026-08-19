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
import random
import re
import time
from typing import Any, Callable

import torch

from swu_env import SWUEnv
from policy import RandomActionPolicy
from runner import EpisodeLogger
from torch_policy import TorchPolicy
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
        verbose: bool = True,
    ):
        self.log_dir = log_dir
        self.champion_path = champion_path
        self.history_dir = os.path.join(log_dir, HISTORY_DIRNAME)
        self.obs_size = obs_size
        self.max_actions = max_actions
        self.device = device
        self.champion_probability = float(champion_probability)
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

    def sample_opponent(self) -> TorchPolicy | None:
        """Sample an opponent policy: champion with `champion_probability`,
        otherwise a random historical checkpoint (falls back to the champion)."""
        if self.has_champion and random.random() < self.champion_probability:
            return self.champion

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
                    return snapshot
                except Exception:
                    pass

        if self.has_champion:
            return self.champion
        return None


# ── Opponent / action helpers ────────────────────────────────────────────────
def _policy_action(policy, env) -> int | None:
    """Sample a masked action index from a TorchPolicy (or a legacy policy)."""
    actions = list(env.available_actions)
    if not actions:
        return None
    if isinstance(policy, TorchPolicy):
        with torch.no_grad():
            obs_vec = torch.tensor(env._get_obs(), dtype=torch.float32)
            idx, _, _ = policy.select_action(obs_vec, actions, getattr(env, "legal_action_mask", None))
        return idx if idx is not None and 0 <= idx < len(actions) else None
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
    env.reset(options=reset_payload)
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
            time.sleep(0.01)
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
            time.sleep(0.01)
            continue

        _, _, terminated, truncated, _ = env.step(action_index)
        if terminated or truncated:
            break

    return _resolve_winner(env)


def run_tournament(
    env,
    candidate,
    champion,
    reset_payload_factory: Callable[[], dict],
    games: int = 50,
    max_steps: int = 1000,
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


# ── Main training loop ───────────────────────────────────────────────────────
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
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Archive candidate weights into the checkpoints history every N episodes")
    parser.add_argument("--stall_polls", type=int, default=40, help="Abort an episode after this many repeated no-action polls in the same prompt state")
    parser.add_argument("--update_every", type=int, default=1, help="Accumulate this many episodes before each policy update (minibatch). 1 = per-episode update, 8-16 recommended for stability")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-episode chatter; keep diagnostics summary lines")
    parser.add_argument("--debug_steps", action="store_true", help="Re-enable verbose per-step action dumps (default: suppressed)")

    # Gated champion pool
    parser.add_argument("--champion_path", type=str, default="", help=f"Champion checkpoint path (default: <log_dir>/{CHAMPION_FILENAME})")
    parser.add_argument("--champion_probability", type=float, default=0.8, help="Fraction of training episodes against the champion (rest: random history checkpoint)")
    parser.add_argument("--tournament_every", type=int, default=500, help="Run the evaluation tournament every N episodes (also after the final episode)")
    parser.add_argument("--tournament_games", type=int, default=50, help="Games per evaluation tournament (P1/P2 seats alternate evenly)")
    parser.add_argument("--promote_win_rate", type=float, default=0.55, help="Candidate win rate threshold for champion promotion")

    # A2C loss coefficients
    parser.add_argument("--value_coef", type=float, default=0.5, help="c1: critic loss weight")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="c2: entropy bonus weight")

    # Diagnostics
    parser.add_argument("--diagnostics_every", type=int, default=20, help="Print the clean summary line every N episodes (clamped to 10-50)")
    parser.add_argument("--no_tensorboard", action="store_true", help="Disable TensorBoard logging even when available")

    args = parser.parse_args()

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
    )

    champion_pool = ChampionPool(
        log_dir=log_dir,
        champion_path=champion_path,
        obs_size=policy.obs_size,
        max_actions=policy.max_actions,
        device=args.device,
        champion_probability=args.champion_probability,
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

    for ep in range(args.episodes):
        episode_number = start_episode + ep + 1
        reset_payload = make_reset_payload()
        obs, info = env.reset(options=reset_payload)
        if verbose:
            print(f"=== Episode {episode_number}/{start_episode + args.episodes} ===")
            logger.log(
                f"[reset] phase={info.get('phase')} activePlayer={info.get('activePlayer')} valid_actions={info.get('num_valid_actions')} prompts={info.get('activePrompts')}",
                player_id=args.player_id,
            )
        logger.record_rl_transition({
            "event": "reset",
            "player_id": args.player_id,
            "state": env.current_state,
            "available_actions": env.available_actions,
            "info": info,
            "decks": fixed_deck_meta if not args.randomize_decks else {},
        })

        # ── Gated opponent sampling: 80% champion / 20% history checkpoint ──
        opponent_policy = champion_pool.sample_opponent()
        opponent = PolicyOpponent(opponent_policy)
        if verbose:
            source = "champion" if opponent_policy is champion_pool.champion else ("history" if opponent_policy is not None else "random")
            print(f"[opponent] episode {episode_number} opponent source: {source}")

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
                    "available_actions": list(env.available_actions),
                    "info": info,
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
                    time.sleep(0.01)
                    continue

                if action >= len(available_actions) or action < 0:
                    logger.log(f"Policy produced invalid action {action} for {len(available_actions)} available", player_id=args.player_id)
                    time.sleep(0.01)
                    continue

                chosen_action = available_actions[action]
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
                    time.sleep(0.01)
                    continue

                opponent_action = list(env.available_actions)[action] if 0 <= action < len(env.available_actions) else None
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
                    step_info = info
                    break

                episode_metrics["total_rewards"] += float(reward)
                episode_metrics["total_reward_steps"] += 1
                episode_metrics["opponent_rewards"] += float(reward)

                if opponent_action and str(opponent_action.get("actionType") or "") == "clickCard":
                    card_uuid = opponent_action.get("uuid", "")
                    opp_state = (state_section.get(opp_key) if opp_key else None) or {}
                    in_hand = any(c.get("uuid") == card_uuid for c in opp_state.get("hand", []))
                    if in_hand:
                        episode_metrics["cards_played"] += 1

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

