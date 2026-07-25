import copy
import json
import threading
import time
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Any, TextIO

from policy import RandomActionPolicy
from swu_env import SWUEnv


@dataclass(frozen=True)
class PlayerSpec:
    player_id: str
    deck_key: str
    label: str


class EpisodeLogger:
    def __init__(self, log_dir: str | None = None, verbose: bool = False) -> None:
        self.verbose = verbose
        self.log_dir = Path(log_dir) if log_dir else None
        self.print_lock = threading.Lock()
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        self.match_handle = None
        self.rl_transition_handle: TextIO | None = None
        self.episode_summary_csv_handle: TextIO | None = None
        self.episode_summary_csv_writer = None
        self.episode_summary_csv_header_written = False
        self.episode_summary_csv_header: list[str] | None = None
        self.step_analysis_csv_handle: TextIO | None = None
        self.step_analysis_csv_writer = None
        self.step_analysis_csv_header_written = False
        self.step_analysis_csv_header: list[str] | None = None
        self._turn_csv_writer = None
        self._turn_csv_handle = None
        self._turn_csv_header = None
        self._turn_csv_header_written = False
        self.player_handles: dict[str, object] = {}

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.match_handle = open(self.log_dir / "match.log", "a", encoding="utf-8", buffering=1)
            self.rl_transition_handle = open(self.log_dir / "rl_transitions.jsonl", "a", encoding="utf-8", buffering=1)
            
            self.episode_summary_csv_handle = open(self.log_dir / "episode_summaries.csv", "a", encoding="utf-8", newline='')
            self.episode_summary_csv_writer = csv.writer(self.episode_summary_csv_handle)

            self.step_analysis_csv_handle = open(self.log_dir / "step_analysis_data.csv", "a", encoding="utf-8", newline='')
            self.step_analysis_csv_writer = csv.writer(self.step_analysis_csv_handle)

    def log(self, message: str, player_id: str | None = None) -> None:
        with self.print_lock:
            if self.verbose or player_id is None:
                print(message)
            if self.match_handle:
                self.match_handle.write(message + "\n")
            if player_id and player_id in self.player_handles:
                self.player_handles[player_id].write(message + "\n")

    def add_player(self, player_id: str) -> None:
        if self.log_dir and player_id not in self.player_handles:
            handle = open(self.log_dir / f"player_{player_id}.log", "a", encoding="utf-8", buffering=1)
            self.player_handles[player_id] = handle

    def record_rl_transition(self, payload: dict[str, Any]) -> None:
        record = {"run_id": self.run_id, **payload}
        with self.print_lock:
            if self.rl_transition_handle:
                self.rl_transition_handle.write(json.dumps(record, default=str) + "\n")

    def record_episode_summary(self, payload: dict[str, Any]) -> None:
        record = {"run_id": self.run_id, **payload}
        with self.print_lock:
            if self.episode_summary_csv_writer:
                if not self.episode_summary_csv_header_written:
                    self.episode_summary_csv_header = list(record.keys())
                    self.episode_summary_csv_writer.writerow(self.episode_summary_csv_header)
                    self.episode_summary_csv_header_written = True
                row = [str(record.get(key, '')) for key in self.episode_summary_csv_header]
                self.episode_summary_csv_writer.writerow(row)

    def record_turn_summary(self, payload: dict[str, Any]) -> None:
        """Log a single-row turn summary to turn_summaries.csv."""
        record = {"run_id": self.run_id, **payload}
        with self.print_lock:
            if not hasattr(self, "_turn_csv_writer") or self._turn_csv_writer is None:
                if self.log_dir:
                    self._turn_csv_handle = open(self.log_dir / "turn_summaries.csv", "a", encoding="utf-8", newline="")
                    self._turn_csv_writer = csv.writer(self._turn_csv_handle)
                    self._turn_csv_header_written = False
                else:
                    return
            if not self._turn_csv_header_written:
                self._turn_csv_header = list(record.keys())
                self._turn_csv_writer.writerow(self._turn_csv_header)
                self._turn_csv_header_written = True
            row = [str(record.get(key, "")) for key in self._turn_csv_header]
            self._turn_csv_writer.writerow(row)

    def record_step_analysis_data(self, payload: dict[str, Any]) -> None:
        record = {"run_id": self.run_id, **payload}
        with self.print_lock:
            if self.step_analysis_csv_writer:
                if not self.step_analysis_csv_header_written:
                    self.step_analysis_csv_header = list(record.keys())
                    self.step_analysis_csv_writer.writerow(self.step_analysis_csv_header)
                    self.step_analysis_csv_header_written = True
                row = [str(record.get(key, '')) for key in self.step_analysis_csv_header]
                self.step_analysis_csv_writer.writerow(row)

    def close(self) -> None:
        for handle in [self.match_handle, self.rl_transition_handle, self.episode_summary_csv_handle, self.step_analysis_csv_handle, self._turn_csv_handle, *self.player_handles.values()]:
            try:
                if handle:
                    handle.close()
            except Exception:
                pass


def print_board_state(state: dict[str, Any] | None) -> None:
    if not state:
        print("<no board state>")
        return

    def _num_value(card: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            value = card.get(key)
            if value is not None:
                return value
        return None

    print("\n[ Board State ]")

    def _print_card(card: dict[str, Any], indent: str) -> None:
        unit_hp = _num_value(card, 'hp', 'remainingHp', 'currentHp')
        unit_power = _num_value(card, 'power', 'printedPower')
        print(f"{indent}- {card.get('internalName', 'Unknown')} (P:{unit_power} HP:{unit_hp} Dmg:{card.get('damage')} Exh:{card.get('exhausted')})")
        upgrades = card.get('upgrades') or []
        for upgrade in upgrades:
            print(f"{indent}  * upgrade: {upgrade.get('internalName', 'Unknown')}")

    for p_key in ["player1", "player2"]:
        if p_key not in state:
            continue
        p = state[p_key]
        print(f"=== {p.get('name', p_key)} ===")

        base = p.get('base', {})
        leader = p.get('leader', {})
        base_hp = _num_value(base, 'hp', 'remainingHp', 'currentHp', 'maxHp')
        leader_hp = _num_value(leader, 'hp', 'remainingHp', 'currentHp', 'maxHp')
        base_power = _num_value(base, 'power', 'printedPower')
        leader_power = _num_value(leader, 'power', 'printedPower')
        base_stats = f"HP: {base_hp}"
        if base_power is not None:
            base_stats += f" P:{base_power}"
        print(f" Base: {base.get('internalName')} ({base_stats}, DMG: {base.get('damage')})")
        leader_stats = f"HP: {leader_hp}"
        if leader_power is not None:
            leader_stats += f" P:{leader_power}"
        print(f" Leader: {leader.get('internalName')} ({leader_stats}, DMG: {leader.get('damage')})")

        space = p.get("spaceArena", [])
        ground = p.get("groundArena", [])

        if space:
            print(" Space Arena:")
            for c in space:
                _print_card(c, "  ")

        if ground:
            print(" Ground Arena:")
            for c in ground:
                _print_card(c, "  ")

        print(f" Resources: {p.get('readyResourceCount')} / {p.get('readyResourceCount', 0) + p.get('exhaustedResourceCount', 0)}")
        print(f" Force Token: {'Yes' if p.get('hasForceToken') else 'No'}")
        print(f" Credit Tokens: {p.get('credits', 0)}")
        print(f" Hand: {len(p.get('hand', []))} cards")


class SingleAgentEpisodeRunner:
    def __init__(self, env: SWUEnv, policy=None, logger: EpisodeLogger | None = None, player_id: str = "111", max_steps: int = 1000) -> None:
        self.env = env
        self.policy = policy or RandomActionPolicy()
        self.logger = logger or EpisodeLogger()
        self.player_id = player_id
        self.max_steps = max_steps
        self.steps = 0

    def run(self, reset_options: dict[str, Any] | None = None) -> None:
        self.logger.log(f"Connecting single-agent runner to {self.env.server_url}")
        self.env.reset(options=reset_options)

        self.logger.record_rl_transition({
                "event": "reset",
                "player_id": self.player_id,
                "step_index": self.steps,
                "state": copy.deepcopy(self.env.current_state),
                "available_actions": copy.deepcopy(self.env.available_actions),
                "reset_options": copy.deepcopy(reset_options),
            }
        )

        terminated = False
        no_action_polls = 0
        last_no_action_signature = None
        while not terminated and self.steps < self.max_steps:
            state_before = copy.deepcopy(self.env.current_state)
            actions_before = copy.deepcopy(self.env.available_actions)
            info_before = self.env._get_info() if hasattr(self.env, '_get_info') else {}
            if self.logger.verbose:
                print_board_state(self.env.current_state.get("state", {}) if self.env.current_state else None)

                prompts = (self.env.current_state or {}).get("prompts") or {}
                for p_key in ("player1", "player2"):
                    p_prompt = prompts.get(p_key) or {}
                    title = p_prompt.get("menuTitle")
                    if title:
                        print(f"prompt {p_key}: {title}")

                actions = self.env.available_actions
                print("Available Actions:")
                for index, action in enumerate(actions):
                    if action.get("actionType") == "clickPrompt":
                        print(f"  [{index}]: button '{action.get('promptText')}'")
                    elif action.get("actionType") == "clickCard":
                        name = action.get('internalName', action.get('uuid'))
                        print(f"  [{index}]: card '{name}'")
                    elif action.get("actionType") == "macro_resource_cards":
                        print(f"  [{index}]: resource cards '{action.get('internalName')}'")
                    else:
                        print(f"  [{index}]: {action}")

            action_index = self.policy.choose_action_index(self.env)
            if action_index is None:
                signature = (
                    str(info_before.get("activePlayer")),
                    str(info_before.get("phase")),
                    len(actions_before),
                    str(((self.env.current_state or {}).get("prompts") or {}).get("player1", {}).get("menuTitle", "")),
                    str(((self.env.current_state or {}).get("prompts") or {}).get("player2", {}).get("menuTitle", "")),
                )
                if signature == last_no_action_signature:
                    no_action_polls += 1
                else:
                    no_action_polls = 1
                    last_no_action_signature = signature

                try:
                    self.env.refresh()
                except Exception as exc:
                    self.logger.log(f"Refresh raised exception while waiting for actions: {exc}", player_id=self.player_id)
                    terminated = True
                    break

                if no_action_polls >= 40:
                    self.logger.log(
                        f"Stalled prompt detected after {no_action_polls} polls with zero legal actions. Aborting episode.",
                        player_id=self.player_id,
                    )
                    self.logger.record_rl_transition({
                            "event": "episode_abort",
                            "player_id": self.player_id,
                            "step_index": self.steps,
                            "state": state_before,
                            "available_actions": actions_before,
                            "info": info_before,
                            "reason": "stalled_no_action",
                            "stall_polls": no_action_polls,
                        }
                    )
                    terminated = True
                    break

                time.sleep(0.05)
                continue

            no_action_polls = 0
            last_no_action_signature = None

            decision = self.policy.describe_choice(self.env, action_index)
            self.logger.log(
                f"[{self.player_id}] step {self.steps + 1}: {decision.action_type} -> {decision.description}",
                player_id=self.player_id,
            )
            try:
                _, reward, terminated, truncated, info = self.env.step(action_index)
            except Exception as exc:
                try:
                    import requests
                    if isinstance(exc, requests.exceptions.HTTPError) and getattr(exc, 'response', None) is not None:
                        body = exc.response.text
                        status = exc.response.status_code
                        self.logger.log(f"HTTPError on step: status={status} body={body}", player_id=self.player_id)
                    else:
                        self.logger.log(f"Step raised exception: {exc}", player_id=self.player_id)
                except Exception:
                    self.logger.log(f"Step raised exception: {exc}", player_id=self.player_id)

                try:
                    self.logger.log(f"Last server state: {self.env.current_state}", player_id=self.player_id)
                except Exception:
                    pass

                terminated = True
                reward = -10.0
                truncated = False
                info = self.env._get_info() if hasattr(self.env, '_get_info') else {}
                self.logger.record_rl_transition({
                        "event": "step_error",
                        "player_id": self.player_id,
                        "step_index": self.steps,
                        "state": state_before,
                        "available_actions": actions_before,
                        "action_index": action_index,
                        "action": copy.deepcopy(actions_before[action_index]) if 0 <= action_index < len(actions_before) else None,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "next_state": copy.deepcopy(self.env.current_state),
                        "info": copy.deepcopy(info),
                        "error": str(exc),
                    }
                )
                self.steps += 1
                self.logger.log(
                    f"[{self.player_id}] result: reward={reward}, terminated={terminated}, truncated={truncated}, phase={info.get('phase')}",
                    player_id=self.player_id,
                )

                if truncated:
                    terminated = True

                time.sleep(0.05)
                continue
            self.steps += 1
            self.logger.record_rl_transition({
                    "event": "step",
                    "player_id": self.player_id,
                    "step_index": self.steps,
                    "state": state_before,
                    "available_actions": actions_before,
                    "action_index": action_index,
                    "action": copy.deepcopy(actions_before[action_index]) if 0 <= action_index < len(actions_before) else None,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "next_state": copy.deepcopy(self.env.current_state),
                    "info": copy.deepcopy(info),
                }
            )
            self.logger.log(
                f"[{self.player_id}] result: reward={reward}, terminated={terminated}, truncated={truncated}, phase={info.get('phase')}",
                player_id=self.player_id,
            )

            if truncated:
                terminated = True

            time.sleep(0.05)


class TwoAgentEpisodeRunner:
    def __init__(self, server_url: str, p1: PlayerSpec, p2: PlayerSpec, max_steps: int = 1000, log_dir: str | None = None, verbose: bool = False) -> None:
        self.server_url = server_url
        self.p1 = p1
        self.p2 = p2
        self.max_steps = max_steps
        self.logger = EpisodeLogger(log_dir=log_dir, verbose=verbose)
        self.envs = {
            p1.player_id: SWUEnv(server_url=server_url, player_id=p1.player_id),
            p2.player_id: SWUEnv(server_url=server_url, player_id=p2.player_id),
        }
        self.policies = {
            p1.player_id: RandomActionPolicy(),
            p2.player_id: RandomActionPolicy(),
        }
        self.terminated = False
        self.steps = 0
        self.condition = threading.Condition()
        self.shared_state = None

        self.logger.add_player(p1.player_id)
        self.logger.add_player(p2.player_id)

    def _broadcast_state(self, state: dict[str, Any]) -> None:
        snapshot = copy.deepcopy(state)
        for env in self.envs.values():
            env.sync_state(copy.deepcopy(snapshot))
        self.shared_state = snapshot

    def reset(self, reset_options: dict[str, Any]) -> None:
        env = self.envs[self.p1.player_id]
        env.reset(options=reset_options)
        self._broadcast_state(env.current_state)
        self.logger.record_rl_transition({
                "event": "reset",
                "player_id": None,
                "step_index": self.steps,
                "state": copy.deepcopy(self.shared_state),
                "available_actions": {
                    pid: copy.deepcopy(self.envs[pid].available_actions)
                    for pid in self.envs
                },
                "reset_options": copy.deepcopy(reset_options),
            }
        )

    def _agent_loop(self, player_id: str) -> None:
        env = self.envs[player_id]
        policy = self.policies[player_id]

        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.terminated or (
                        self.shared_state is not None and self.shared_state.get("activePlayer") == player_id and len(env.available_actions) > 0
                    )
                )
                if self.terminated:
                    return

                action_index = policy.choose_action_index(env)
                if action_index is None:
                    self.condition.wait(timeout=0.05)
                    continue

                decision = policy.describe_choice(env, action_index)
                state_before = copy.deepcopy(env.current_state)
                actions_before = copy.deepcopy(env.available_actions)
                self.logger.log(
                    f"[{player_id}] choosing {action_index}: {decision.action_type} {decision.description}",
                    player_id=player_id,
                )

            try:
                _, reward, terminated, truncated, info = env.step(action_index)
            except Exception as exc:
                with self.condition:
                    self.terminated = True
                    self.logger.log(f"[{player_id}] step failed: {exc}", player_id=player_id)
                    self.logger.record_rl_transition({
                            "event": "step_error",
                            "player_id": player_id,
                            "step_index": self.steps,
                            "state": state_before,
                            "available_actions": actions_before,
                            "action_index": action_index,
                            "action": copy.deepcopy(actions_before[action_index]) if 0 <= action_index < len(actions_before) else None,
                            "reward": -10.0,
                            "terminated": True,
                            "truncated": False,
                            "next_state": copy.deepcopy(env.current_state),
                            "info": copy.deepcopy(env._get_info()) if hasattr(env, '_get_info') else {},
                            "error": str(exc),
                        }
                    )
                    self.condition.notify_all()
                return

            with self.condition:
                self.steps += 1
                self._broadcast_state(env.current_state)
                self.logger.record_rl_transition({
                        "event": "step",
                        "player_id": player_id,
                        "step_index": self.steps,
                        "state": state_before,
                        "available_actions": actions_before,
                        "action_index": action_index,
                        "action": copy.deepcopy(actions_before[action_index]) if 0 <= action_index < len(actions_before) else None,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "next_state": copy.deepcopy(env.current_state),
                        "info": copy.deepcopy(info),
                    }
                )
                self.logger.log(
                    f"[{player_id}] step result: reward={reward}, terminated={terminated}, truncated={truncated}, phase={info.get('phase')}",
                    player_id=player_id,
                )
                if terminated or truncated or self.steps >= self.max_steps:
                    self.terminated = True
                self.condition.notify_all()

            time.sleep(0.05)

    def run(self, reset_options: dict[str, Any]) -> None:
        try:
            self.reset(reset_options)
            threads = [
                threading.Thread(target=self._agent_loop, args=(self.p1.player_id,), daemon=True),
                threading.Thread(target=self._agent_loop, args=(self.p2.player_id,), daemon=True),
            ]
            for thread in threads:
                thread.start()
            while any(thread.is_alive() for thread in threads):
                for thread in threads:
                    thread.join(timeout=0.25)
        finally:
            self.logger.close()
