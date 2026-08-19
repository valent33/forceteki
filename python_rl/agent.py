import argparse
import hashlib
import importlib
import itertools
import os
import json
import random
import re
import threading
import time
from typing import Any
from urllib.parse import quote_plus

import requests
import socketio
import torch  # type: ignore[import-not-found]
from swu_env import SWUEnv, load_card_database
from deck_utils import load_deck


QUEUE_FORMAT = "premier"
QUEUE_CARD_POOL = "current"
QUEUE_GAMES_TO_WIN = "bestOfOne"


def _resolve_relative_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _bucketed_hash_features(text: str, size: int) -> list[float]:
    buckets = [0.0] * size
    if not text or size <= 0:
        return buckets

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    for index in range(size):
        buckets[index] = float(digest[index % len(digest)] / 255.0)
    return buckets


def _load_policy_checkpoint(torch_module: Any, policy: Any, checkpoint_path: str, device: str) -> dict[str, Any]:
    checkpoint = torch_module.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        policy.net.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            policy.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return {key: value for key, value in checkpoint.items() if key not in {"model_state_dict", "optimizer_state_dict"}}

    policy.net.load_state_dict(checkpoint)
    return {}


def _load_optional_opponent_deck(deck_key, decks_file):
    if not deck_key:
        return None
    leader, base, deck = load_deck(deck_key, decks_file)
    return {
        "leader": leader,
        "base": base,
        "cards": deck,
    }


def _load_raw_deck(deck_key: str, decks_file: str) -> dict[str, Any]:
    with open(decks_file, "r", encoding="utf-8") as file_handle:
        decks_db = json.load(file_handle)

    if deck_key not in decks_db:
        raise KeyError(f"Deck '{deck_key}' was not found in {decks_file}")

    return decks_db[deck_key]


def _normalize_server_url(server_url: str) -> str:
    if server_url.startswith("http://") or server_url.startswith("https://"):
        return server_url.rstrip("/")
    return f"http://{server_url.rstrip('/')}"


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


# ─────────────────────────────────────────────────────────────────────────────
# Board visualizer (--debug_vis)
# ─────────────────────────────────────────────────────────────────────────────
_CARD_DB_REF: dict[str, dict[str, Any]] | None = None


def _card_def(internal_name: Any) -> dict[str, Any]:
    """Card-database entry for an internal name (cached card database)."""
    global _CARD_DB_REF
    if _CARD_DB_REF is None:
        try:
            _CARD_DB_REF = load_card_database()
        except Exception:
            _CARD_DB_REF = {}
    name = str(internal_name or "").strip().lower()
    if not name:
        return {}
    return _CARD_DB_REF.get(name, {})


def _num(card: Any, *keys: str, default=None):
    if not isinstance(card, dict):
        return default
    for key in keys:
        value = card.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _format_hp(hp, max_hp):
    if hp is None:
        return "?"
    if max_hp is None:
        return f"{hp:.0f}"
    return f"{hp:.0f}/{max_hp:.0f}"


def _build_rl_view(env) -> dict[str, Any]:
    """Normalized board view from the RL env-server state format."""
    state = env.current_state or {}
    section = state.get("state") or {}
    player_key = "player1" if str(state.get("player1Id")) == str(env.player_id) else "player2"
    opp_key = "player2" if player_key == "player1" else "player1"

    def player_view(p_key: str) -> dict[str, Any]:
        pstate = section.get(p_key) or {}
        base = pstate.get("base") or {}
        leader = pstate.get("leader") or {}
        base_db = _card_def(base.get("internalName"))
        leader_db = _card_def(leader.get("internalName"))
        leader_zone = str(leader.get("zone") or "").lower()

        def unit_row(card: dict[str, Any]) -> dict[str, Any]:
            db = _card_def(card.get("internalName"))
            upgrades = card.get("upgrades") or []
            shields = sum(1 for u in upgrades if str(u.get("internalName", "")).lower() == "shield")
            keywords = {str(k).lower() for k in (db.get("keywords") or [])}
            hp = _num(card, "hp", "currentHp", default=0.0)
            max_hp = db.get("hp") if db.get("hp") is not None else hp + (_num(card, "damage", default=0.0))
            return {
                "name": str(db.get("title") or card.get("internalName") or "?"),
                "power": _num(card, "power", "printedPower", default=0.0),
                "hp": hp,
                "max_hp": max_hp,
                "shield": shields,
                "sentinel": "sentinel" in keywords,
                "exhausted": bool(card.get("exhausted") or card.get("isExhausted") or card.get("is_exhausted")),
                "upgrades": [
                    str(u.get("internalName", "?"))
                    for u in upgrades
                    if str(u.get("internalName", "")).lower() != "shield"
                ],
            }

        ground = [unit_row(c) for c in (pstate.get("groundArena") or [])]
        space = [unit_row(c) for c in (pstate.get("spaceArena") or [])]
        if "ground" in leader_zone:
            ground.append(unit_row(leader))
        elif "space" in leader_zone:
            space.append(unit_row(leader))

        hand_cards = pstate.get("hand") or []
        return {
            "name": str(pstate.get("name") or p_key),
            "base": {
                "name": str(base_db.get("title") or base.get("internalName") or "?"),
                "hp": _num(base, "hp", "currentHp", default=0.0),
                "max_hp": base_db.get("hp") if base_db.get("hp") is not None else None,
            },
            "leader_name": str(leader_db.get("title") or leader.get("internalName") or "?"),
            "leader_deployed": bool("ground" in leader_zone or "space" in leader_zone),
            "ground": ground,
            "space": space,
            "hand": [str(_card_def(c.get("internalName")).get("title") or c.get("internalName") or "?") for c in hand_cards],
            "hand_hidden": 0,
            "deck_count": len(pstate.get("deck") or []),
            "resources_ready": _num(pstate, "readyResourceCount", default=0.0),
            "resources_total": _num(pstate, "readyResourceCount", default=0.0) + _num(pstate, "exhaustedResourceCount", default=0.0),
            "credits": _num(pstate, "credits", default=0.0),
            "force": bool(pstate.get("hasForceToken")),
        }

    mask = getattr(env, "legal_action_mask", None)
    legal_total = len(env.available_actions)
    legal_unmasked = int(mask.sum()) if mask is not None and len(mask) >= legal_total else legal_total

    return {
        "phase": str(state.get("phase") or "?"),
        "active_player": "ME" if str(state.get("activePlayer") or "") == str(env.player_id) else "OPP",
        "friendly": player_view(player_key),
        "enemy": player_view(opp_key),
        "legal_total": legal_total,
        "legal_unmasked": legal_unmasked,
        "source": "rl",
    }


def _build_gui_view(state: dict[str, Any], player_id: str) -> dict[str, Any]:
    """Normalized board view from the GUI game-state format (queue mode)."""
    players = state.get("players") or {}
    my_state = players.get(player_id)
    enemy_state = next(
        (pstate for pid, pstate in players.items() if str(pid) != str(player_id) and isinstance(pstate, dict)),
        None,
    )

    def player_view(pstate, is_me: bool) -> dict[str, Any]:
        empty = {
            "name": "?",
            "base": {"name": "?", "hp": 0.0, "max_hp": None},
            "leader_name": "?",
            "leader_deployed": False,
            "ground": [],
            "space": [],
            "hand": [],
            "hand_hidden": 0,
            "deck_count": 0,
            "resources_ready": 0.0,
            "resources_total": None,
            "credits": 0,
            "force": False,
        }
        if not isinstance(pstate, dict):
            return empty

        piles = pstate.get("cardPiles") or {}

        def unit_row(card: dict[str, Any]) -> dict[str, Any]:
            return {
                "name": str(card.get("name") or "?"),
                "power": _num(card, "power", default=0.0),
                "hp": _num(card, "hp", default=0.0),
                "max_hp": None,
                "shield": None,
                "sentinel": bool(card.get("sentinel")),
                "exhausted": bool(card.get("exhausted")),
                "upgrades": [],
            }

        ground = [unit_row(c) for c in (piles.get("groundArena") or []) if isinstance(c, dict)]
        space = [unit_row(c) for c in (piles.get("spaceArena") or []) if isinstance(c, dict)]
        leader = pstate.get("leader") or {}
        leader_zone = str(leader.get("zone") or "").lower()
        if "ground" in leader_zone:
            ground.append(unit_row(leader))
        elif "space" in leader_zone:
            space.append(unit_row(leader))

        hand_cards = [c for c in (piles.get("hand") or []) if isinstance(c, dict)]
        hand_names = [str(c.get("name") or "?") for c in hand_cards if c.get("name")]
        base = pstate.get("base") or {}
        credits_list = pstate.get("credits") or piles.get("credits") or []
        force_token = pstate.get("forceToken") or {}

        return {
            "name": str(pstate.get("name") or ("ME" if is_me else "OPP")),
            "base": {
                "name": str(base.get("name") or "?"),
                "hp": _num(base, "hp", default=0.0),
                "max_hp": None,
            },
            "leader_name": str(leader.get("name") or "?"),
            "leader_deployed": bool("ground" in leader_zone or "space" in leader_zone),
            "ground": ground,
            "space": space,
            "hand": hand_names,
            "hand_hidden": max(0, len(hand_cards) - len(hand_names)),
            "deck_count": int(pstate.get("numCardsInDeck") or 0),
            "resources_ready": _num(pstate, "availableResources", default=0.0),
            "resources_total": None,
            "credits": len(credits_list),
            "force": bool(force_token.get("active")),
        }

    prompt_state = (my_state or {}).get("promptState") or {}
    legal_total = (
        len(prompt_state.get("buttons") or [])
        + len(prompt_state.get("dropdownListOptions") or [])
        + len(prompt_state.get("displayCards") or [])
        + len(prompt_state.get("perCardButtons") or [])
    )
    if isinstance(my_state, dict):
        piles = my_state.get("cardPiles") or {}
        for zone_cards in piles.values():
            if not isinstance(zone_cards, list):
                continue
            for card in zone_cards:
                if isinstance(card, dict) and card.get("selectable"):
                    legal_total += 1
        for key in ("leader", "base"):
            card = my_state.get(key)
            if isinstance(card, dict) and card.get("selectable"):
                legal_total += 1

    return {
        "phase": str(state.get("phase") or "?"),
        "active_player": "ME" if isinstance(my_state, dict) and my_state.get("isActionPhaseActivePlayer") else "OPP",
        "friendly": player_view(my_state, True),
        "enemy": player_view(enemy_state, False),
        "legal_total": legal_total,
        "legal_unmasked": legal_total,
        "source": "gui",
    }


def _render_board_view(view: dict[str, Any]) -> None:
    friendly = view["friendly"]
    enemy = view["enemy"]
    fb, eb = friendly["base"], enemy["base"]

    print("═" * 76)
    print(f" SWU BOARD  |  Phase: {view['phase']}  |  Active: {view['active_player']}")
    print(f" Base HP     ME {fb['name']} {_format_hp(fb['hp'], fb['max_hp'])}"
          f"   |   OPP {eb['name']} {_format_hp(eb['hp'], eb['max_hp'])}")
    print(f" Leaders     ME {friendly['leader_name']} (deployed={'yes' if friendly['leader_deployed'] else 'no'})"
          f"   |   OPP {enemy['leader_name']} (deployed={'yes' if enemy['leader_deployed'] else 'no'})")

    def _res_line(side) -> str:
        ready = side["resources_ready"]
        total = side["resources_total"]
        if total is None:
            return f"{ready:.0f}"
        return f"{ready:.0f}/{total:.0f}"

    force_holder = "ME" if friendly["force"] else ("OPP" if enemy["force"] else "—")
    print(f" Resources   ME {_res_line(friendly)}  |  OPP {_res_line(enemy)}"
          f"      Credits  ME {friendly['credits']:.0f}  OPP {enemy['credits']:.0f}"
          f"      Force: {force_holder}")

    def _print_arena(title: str, my_units, opp_units) -> None:
        print(f" {title}:")
        if not my_units and not opp_units:
            print("    (empty)")
            return
        for owner_label, units in (("ME ", my_units), ("OPP", opp_units)):
            for unit in units:
                flags = [f"P {unit['power']:.0f}", f"HP {_format_hp(unit['hp'], unit['max_hp'])}"]
                if unit.get("shield"):
                    flags.append(f"shield {unit['shield']}")
                if unit.get("sentinel"):
                    flags.append("SENTINEL")
                if unit.get("exhausted"):
                    flags.append("exhausted")
                upgrades = ", ".join(unit.get("upgrades") or [])
                line = f"    {owner_label} {unit['name']:<30} {' '.join(flags)}"
                if upgrades:
                    line += f"  [+{upgrades}]"
                print(line)

    _print_arena("Ground Arena", friendly["ground"], enemy["ground"])
    _print_arena("Space Arena", friendly["space"], enemy["space"])

    def _hand_line(side) -> str:
        if not side["hand"]:
            return "—"
        names = side["hand"][:8]
        return ", ".join(names) + (" …" if len(side["hand"]) > 8 else "")

    enemy_hand_count = len(enemy["hand"]) + enemy["hand_hidden"]
    enemy_belief = f" ({_hand_line(enemy)})" if enemy["hand"] else ""
    print(f" Hand        ME [{_hand_line(friendly)}]  |  OPP {enemy_hand_count} cards{enemy_belief}")
    print(f" Deck        ME {friendly['deck_count']} cards  |  OPP {enemy['deck_count']} cards")

    if view.get("legal_total") is not None:
        print(f" Legal Actions  total={view['legal_total']}  unmasked={view['legal_unmasked']}")
    print("═" * 76)


def render_rl_board(env) -> None:
    """Print the --debug_vis board from the RL env-server state."""
    try:
        _render_board_view(_build_rl_view(env))
    except Exception as exc:  # never let the visualizer crash the agent
        print(f"[debug_vis] render failed: {exc}")


def render_gui_board(state: dict[str, Any], player_id: str) -> None:
    """Print the --debug_vis board from the GUI game state."""
    try:
        _render_board_view(_build_gui_view(state, player_id))
    except Exception as exc:  # never let the visualizer crash the agent
        print(f"[debug_vis] render failed: {exc}")


class QueueBotClient:
    def __init__(
        self,
        server_url: str,
        deck_key: str,
        decks_file: str,
        player_id: str,
        max_steps: int,
        policy_checkpoint: str | None = None,
        display_name: str | None = None,
        log_handle=None,
        console_logging: bool = True,
        verbose: bool = False,
        debug_vis: bool = False,
    ) -> None:
        self.server_url = _normalize_server_url(server_url)
        self.player_id = player_id
        self.deck_key = deck_key
        self.max_steps = max_steps
        self.policy_checkpoint = _resolve_relative_path(policy_checkpoint) if policy_checkpoint else None
        self.display_name = display_name or f"python-bot-{self.player_id}"
        self.log_handle = log_handle
        self.console_logging = console_logging
        self.verbose = verbose
        self.debug_vis = debug_vis
        self.deck_payload = _load_raw_deck(deck_key, decks_file)
        self.socket = socketio.Client(reconnection=True, logger=False, engineio_logger=False)
        self.current_state: dict[str, Any] | None = None
        self.last_state_signature: str | None = None
        self.last_action_description: str | None = None
        self.last_prompt_uuid: str | None = None
        self.last_action_kind: str | None = None
        self.game_over = False
        self.game_over_reason: str | None = None
        self.seen_ongoing_game = False
        self.steps_taken = 0
        self.action_lock = threading.Lock()
        self.policy: Any = None
        if self.policy_checkpoint:
            if not os.path.exists(self.policy_checkpoint):
                raise FileNotFoundError(f"Policy checkpoint not found: {self.policy_checkpoint}")
            try:
                torch_module = importlib.import_module("torch")
                torch_policy_module = importlib.import_module("torch_policy")
                policy_class = getattr(torch_policy_module, "TorchPolicy")
                # Derive the architecture from the checkpoint itself so both the
                # DualHeadNetwork checkpoints (2386-dim) and legacy ones load.
                ckpt_data = torch_module.load(self.policy_checkpoint, map_location="cpu")
                state_dict = ckpt_data.get("model_state_dict", ckpt_data)
                ckpt_obs_size = (
                    state_dict.get("obs_encoder.0.weight").shape[1]
                    if "obs_encoder.0.weight" in state_dict
                    else 2386
                )
                ckpt_max_actions = (
                    state_dict.get("policy_head.weight").shape[0]
                    if "policy_head.weight" in state_dict
                    else 100
                )
                self.policy = policy_class(
                    obs_size=ckpt_obs_size,
                    max_actions=ckpt_max_actions,
                    device="cpu",
                )
                metadata = _load_policy_checkpoint(torch_module, self.policy, self.policy_checkpoint, "cpu")
                self._log(f"Loaded policy checkpoint from {self.policy_checkpoint} (obs={ckpt_obs_size}, actions={ckpt_max_actions})")
                if metadata:
                    self._log(f"Policy checkpoint metadata: {metadata}")
            except ImportError as exc:
                self._log(f"Policy checkpoint disabled because the dependency could not be imported: {exc}")
                self.policy = None
        self.user_payload = {
            "id": self.player_id,
            "username": self.display_name,
        }
        self._register_handlers()

    def _log(self, message: str) -> None:
        if self.console_logging:
            print(message)
        if self.log_handle:
            self.log_handle.write(message + "\n")

    def _register_handlers(self) -> None:
        @self.socket.event
        def connect():
            self._log(f"Connected to {self.server_url} as {self.user_payload['username']}")

        @self.socket.event
        def disconnect():
            if self.game_over:
                self._log(f"Socket disconnected after game end: {self.game_over_reason or 'unknown reason'}")
            else:
                self._log("Socket disconnected")

        @self.socket.on("connection_error")
        def connection_error(error):
            self._log(f"Connection error: {error}")

        @self.socket.on("matchmakingFailed")
        def matchmaking_failed(error):
            self._log(f"Matchmaking failed: {error}")

        @self.socket.on("inactiveDisconnect")
        def inactive_disconnect():
            self._log("Disconnected due to inactivity")

        @self.socket.on("queueHeartbeat")
        def queue_heartbeat(*args, **kwargs):
            return None

        @self.socket.on("lobbystate")
        def lobby_state(lobby_state):
            if isinstance(lobby_state, dict):
                self._log(f"Lobby state updated: {lobby_state.get('id', 'unknown')}")
                if lobby_state.get("gameOngoing"):
                    self.seen_ongoing_game = True
                self._maybe_finish_from_lobby_state(lobby_state)
                if self.game_over:
                    return

        @self.socket.on("gamestate")
        def game_state(game_state):
            self.current_state = game_state
            phase = game_state.get("phase") if isinstance(game_state, dict) else None
            game_id = game_state.get("id") if isinstance(game_state, dict) else None
            self._log(f"Received gamestate: phase={phase}, game_id={game_id}")
            if self.debug_vis and isinstance(game_state, dict):
                render_gui_board(game_state, self.player_id)
            if isinstance(game_state, dict):
                self._log_game_state(game_state)
            self._maybe_finish_from_game_state(game_state)
            if self.last_action_description:
                self._log(f"[result] phase={phase}, last_action={self.last_action_description}")
                self.last_action_description = None
            if self.game_over:
                return
            self._maybe_take_action()

    def connect_and_queue(self) -> None:
        socket_query = {
            "user": json.dumps(self.user_payload),
            "lobby": json.dumps({"lobbyId": None}),
            "spectator": "false",
        }
        queue_url = f"{self.server_url}/api/enter-queue"
        connect_url = f"{self.server_url}?{self._encode_query(socket_query)}"

        queue_payload = {
            "user": self.user_payload,
            "deck": self.deck_payload,
            "format": QUEUE_FORMAT,
            "cardPool": QUEUE_CARD_POOL,
            "gamesToWinMode": QUEUE_GAMES_TO_WIN,
        }

        self._log(f"Joining queue via {queue_url}")
        response = requests.post(queue_url, json=queue_payload, timeout=30)
        if not response.ok:
            self._log(f"Queue request failed with status {response.status_code}: {response.text}")
            response.raise_for_status()
        self._log("Queue entry accepted")

        self._log(f"Connecting socket to {connect_url}")
        self.socket.connect(connect_url, socketio_path="ws", transports=["websocket", "polling"])

    def wait(self) -> None:
        try:
            while self.socket.connected and not self.game_over:
                self.socket.sleep(0.25)
        except KeyboardInterrupt:
            self._log("Interrupted, disconnecting socket")
            self.socket.disconnect()
            raise
        finally:
            if self.game_over and self.socket.connected:
                self.socket.disconnect()

    @staticmethod
    def _encode_query(query: dict[str, str]) -> str:
        return "&".join(f"{key}={quote_plus(value)}" for key, value in query.items())

    def _maybe_take_action(self) -> None:
        with self.action_lock:
            if self.game_over:
                return
            state = self.current_state
            if not state or "players" not in state:
                return

            player_state = state["players"].get(self.player_id)
            if not player_state:
                return

            prompt_state = player_state.get("promptState") or {}
            prompt_uuid = prompt_state.get("promptUuid")
            if not prompt_uuid:
                return

            state_signature = json.dumps(player_state, sort_keys=True, default=str)
            if state_signature == self.last_state_signature:
                return

            self._log_prompt_state(player_state, prompt_state)
            action = self._choose_action(state, prompt_state)
            if not action:
                self._log(f"[{self.player_id}] no action chosen for prompt {prompt_uuid}")
                return

            self._emit_action(action)
            self.last_state_signature = state_signature
            self.last_action_description = action.get('description', 'unknown')
            self.last_action_kind = action.get('kind')
            self.last_prompt_uuid = prompt_uuid
            self.steps_taken += 1
            self._log(f"[{self.player_id}] step {self.steps_taken}: {action['actionType']} -> {action.get('description', 'unknown')}")
            if self.steps_taken >= self.max_steps:
                self._log(f"Reached max_steps={self.max_steps}, disconnecting socket")
                self.socket.disconnect()

    def _log_prompt_state(self, player_state: dict[str, Any], prompt_state: dict[str, Any]) -> None:
        prompt_type = prompt_state.get("promptType") or "unknown"
        menu_title = prompt_state.get("menuTitle") or ""
        candidates = self._build_candidates(self.current_state or {}, prompt_state)

        self._log(
            f"[{self.player_id}] prompt: type={prompt_type} title={menu_title!r} "
            f"candidates={len(candidates)}"
        )

        if candidates:
            self._log(f"[{self.player_id}] Available Actions:")
            for index, candidate in enumerate(candidates[:20]):
                self._log(f"[{self.player_id}]   [{index}]: {candidate.get('kind')} '{candidate.get('description', 'unknown')}'")

    def _log_game_state(self, game_state: dict[str, Any]) -> None:
        if not self.verbose:
            return
        board_state = game_state.get("state")
        if board_state is None:
            board_state = game_state.get("players")
        print_board_state(board_state)

    def _choose_action(self, state: dict[str, Any], prompt_state: dict[str, Any]) -> dict[str, Any] | None:
        import time
        time.sleep(0.5)
        
        prompt_type = prompt_state.get("promptType")
        prompt_uuid = prompt_state.get("promptUuid")
        prompt_title = str(prompt_state.get("menuTitle") or "").lower()

        if prompt_type == "distributeAmongTargets" and prompt_state.get("distributeAmongTargets"):
            distribution = self._build_distribution_results(state, prompt_state)
            if distribution is None:
                return None
            return {
                "actionType": "statefulPromptResults",
                "uuid": prompt_uuid,
                "result": distribution,
                "description": f"statefulPromptResults {distribution.get('type')}",
            }

        candidates = self._build_candidates(state, prompt_state)
        if candidates:
            if prompt_state.get("promptType") == "resource":
                resource_candidates = [candidate for candidate in candidates if candidate.get("kind") == "macro_resource_cards"]
                if resource_candidates:
                    return self._choose_policy_candidate(state, prompt_state, resource_candidates)

            if prompt_type == "displayCards":
                done_candidates = [
                    candidate for candidate in candidates
                    if candidate.get("kind") == "clickPrompt" and str(candidate.get("description", "")).lower() == "done"
                ]
                if done_candidates and self._should_acknowledge_display_prompt(prompt_title):
                    return self._choose_policy_candidate(state, prompt_state, done_candidates)

            card_like_candidates = [
                candidate for candidate in candidates
                if candidate.get("kind") in {"clickCard", "perCardMenuButton", "statefulPromptResults"}
                or (candidate.get("kind") == "clickPrompt" and str(candidate.get("description", "")).lower().startswith("select card"))
                or (candidate.get("kind") == "clickPrompt" and str(candidate.get("description", "")).lower().startswith("dropdown"))
            ]
            if card_like_candidates:
                return self._choose_policy_candidate(state, prompt_state, card_like_candidates)

            button_candidates = [
                candidate for candidate in candidates
                if candidate.get("kind") in {"clickPrompt", "button"}
            ]
            non_pass_button_candidates = [candidate for candidate in button_candidates if not self._is_pass_like_candidate(candidate)]
            if non_pass_button_candidates:
                button_candidates = non_pass_button_candidates
            if button_candidates:
                return self._choose_policy_candidate(state, prompt_state, button_candidates)

            return self._choose_policy_candidate(state, prompt_state, candidates)

        self._log(f"[{self.player_id}] no supported action found for prompt type {prompt_type!r}")
        return None

    def _should_acknowledge_display_prompt(self, prompt_title: str) -> bool:
        if not prompt_title:
            return False

        if prompt_title.startswith("view cards"):
            return True

        if any(keyword in prompt_title for keyword in ("reveal", "revealed", "look at", "show cards", "display cards")):
            return True

        return False

    def _build_candidates(self, state: dict[str, Any], prompt_state: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_uuid = prompt_state.get("promptUuid")
        candidates: list[dict[str, Any]] = []

        display_cards = [card for card in (prompt_state.get("displayCards") or []) if card.get("selectionState") != "invalid"]
        per_card_buttons = [button for button in (prompt_state.get("perCardButtons") or []) if not button.get("disabled", False)]
        buttons = [button for button in (prompt_state.get("buttons") or []) if not button.get("disabled", False)]
        dropdown_options = prompt_state.get("dropdownListOptions") or []
        selectable_cards = self._collect_selectable_cards(state)
        selected_cards = prompt_state.get("selectedCards") or []

        if prompt_state.get("promptType") == "resource":
            resource_candidates = self._build_resource_candidates(prompt_state, selectable_cards)
            if resource_candidates:
                candidates.extend(resource_candidates)

            for button in buttons:
                if str(button.get("arg", "")).strip().lower() == "done" or str(button.get("text", "")).strip().lower() == "done":
                    candidates.append({
                        "kind": "clickPrompt",
                        "actionType": "clickPrompt",
                        "arg": button.get("arg", "done"),
                        "uuid": prompt_uuid,
                        "method": button.get("command") or "menuButton",
                        "description": "done",
                    })

            return self._annotate_candidates(state, prompt_state, candidates)

        if prompt_state.get("promptType") == "distributeAmongTargets" and prompt_state.get("distributeAmongTargets"):
            distribution = self._build_distribution_results(state, prompt_state)
            if distribution is not None:
                candidates.append({
                    "kind": "statefulPromptResults",
                    "actionType": "clickPrompt",
                    "uuid": prompt_uuid,
                    "result": distribution,
                    "description": f"statefulPromptResults {distribution.get('type')}",
                })
            return self._annotate_candidates(state, prompt_state, candidates)

        if display_cards:
            if per_card_buttons:
                for card in display_cards:
                    for button in per_card_buttons:
                        candidates.append({
                            "kind": "perCardMenuButton",
                            "actionType": "clickPrompt",
                            "arg": button.get("arg", ""),
                            "cardUuid": card.get("cardUuid"),
                            "uuid": prompt_uuid,
                            "method": button.get("command") or "perCardMenuButton",
                            "description": f"{button.get('text', 'button')} on {card.get('internalName', card.get('cardUuid'))}",
                        })
            else:
                for card in display_cards:
                    candidates.append({
                        "kind": "menuButton",
                        "actionType": "clickPrompt",
                        "arg": card.get("cardUuid"),
                        "uuid": prompt_uuid,
                        "method": "menuButton",
                        "description": f"select card {card.get('cardUuid')}",
                    })

        if prompt_state.get("promptType") == "resource" and selected_cards:
            selected_uuids = {card.get("uuid") for card in selected_cards if isinstance(card, dict)}
            selectable_cards = [card for card in selectable_cards if card.get("uuid") not in selected_uuids]

        for card in selectable_cards:
            candidates.append({
                "kind": "cardClicked",
                "actionType": "clickCard",
                "cardUuid": card.get("uuid"),
                "description": card.get("internalName", card.get("uuid", "card")),
            })

        for option in dropdown_options:
            candidates.append({
                "kind": "menuButton",
                "actionType": "clickPrompt",
                "arg": option,
                "uuid": prompt_uuid,
                "method": "menuButton",
                "description": f"dropdown {option}",
            })

        for button in buttons:
            command = button.get("command") or "menuButton"
            candidates.append({
                "kind": command,
                "actionType": "clickPrompt",
                "arg": button.get("arg", ""),
                "uuid": prompt_uuid,
                "method": command,
                "description": button.get("text", "button"),
            })

        return self._annotate_candidates(state, prompt_state, candidates)

    def _build_resource_candidates(self, prompt_state: dict[str, Any], selectable_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        resource_count = self._resource_card_count(prompt_state)
        if resource_count is None or resource_count <= 0:
            return []

        resource_cards = [card for card in selectable_cards if isinstance(card, dict) and card.get("uuid")]
        if len(resource_cards) < resource_count:
            return []

        candidates: list[dict[str, Any]] = []
        for card_group in itertools.combinations(resource_cards, resource_count):
            uuids = [str(card.get("uuid")) for card in card_group]
            names = [str(card.get("internalName", card.get("uuid", "card"))) for card in card_group]
            candidates.append({
                "kind": "macro_resource_cards",
                "actionType": "macro_resource_cards",
                "uuids": uuids,
                "arg": "any",
                "uuid": prompt_state.get("promptUuid"),
                "method": "macro_resource_cards",
                "description": " + ".join(names),
                "features": self._build_resource_candidate_features(card_group),
            })

        return candidates

    @staticmethod
    def _resource_card_count(prompt_state: dict[str, Any]) -> int | None:
        menu_title = str(prompt_state.get("menuTitle") or prompt_state.get("promptTitle") or "").lower()
        match = re.search(r"select\s+(\d+)\s+cards?\s+to\s+resource", menu_title)
        if match:
            return int(match.group(1))

        if "resource" in menu_title:
            return 1

        return None

    def _build_resource_candidate_features(self, cards: tuple[dict[str, Any], ...]) -> dict[str, float]:
        first_card = cards[0] if cards else {}
        card_type = str(first_card.get("cardType") or "").lower()
        power_val = float(first_card.get("power") or first_card.get("printedPower") or 0.0)
        hp_val = float(first_card.get("hp") or first_card.get("remainingHp") or first_card.get("currentHp") or 0.0)

        return {
            "is_stateful": 0.0,
            "is_macro": 1.0,
            "is_dropdown": 0.0,
            "is_done": 0.0,
            "is_card": 0.0,
            "is_friendly": 1.0,
            "is_leader": 1.0 if card_type == "leader" else 0.0,
            "is_base": 1.0 if card_type == "base" else 0.0,
            "is_exhausted": 1.0 if first_card.get("exhausted") else 0.0,
            "is_unit": 1.0 if card_type == "unit" else 0.0,
            "card_power": power_val / 10.0,
            "card_hp": hp_val / 20.0,
        }

    @staticmethod
    def _is_pass_like_candidate(candidate: dict[str, Any]) -> bool:
        description = str(candidate.get("description", "")).strip().lower()
        prompt_text = str(candidate.get("promptText", "")).strip().lower()
        arg = str(candidate.get("arg", "")).strip().lower()
        return description in {"pass", "done"} or prompt_text in {"pass", "done"} or arg in {"pass", "done"}

    def _annotate_candidates(self, state: dict[str, Any], prompt_state: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for candidate in candidates:
            candidate["features"] = self._build_candidate_features(state, prompt_state, candidate)
        return candidates

    def _build_candidate_features(self, state: dict[str, Any], prompt_state: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
        candidate_kind = str(candidate.get("kind", ""))
        description = str(candidate.get("description", "")).lower()
        card_uuid = candidate.get("cardUuid") or candidate.get("arg")
        card = self._find_card(state, card_uuid)

        features: dict[str, float] = {
            "is_stateful": 1.0 if candidate_kind == "statefulPromptResults" else 0.0,
            "is_macro": 0.0,
            "is_dropdown": 1.0 if description.startswith("dropdown") else 0.0,
            "is_done": 1.0 if description == "done" else 0.0,
            "is_card": 1.0 if candidate_kind in {"cardClicked", "perCardMenuButton"} or description.startswith("select card") else 0.0,
            "is_friendly": 0.0,
            "is_leader": 0.0,
            "is_base": 0.0,
            "is_exhausted": 0.0,
            "is_unit": 0.0,
            "card_power": 0.0,
            "card_hp": 0.0,
        }

        if card:
            features["is_friendly"] = 1.0 if card[1] else 0.0
            card_data = card[0]
            features["is_leader"] = 1.0 if card_data.get("cardType") == "leader" else 0.0
            features["is_base"] = 1.0 if card_data.get("cardType") == "base" else 0.0
            features["is_exhausted"] = 1.0 if card_data.get("exhausted") else 0.0
            features["is_unit"] = 1.0 if card_data.get("cardType") == "unit" else 0.0
            features["card_power"] = float(card_data.get("power") or card_data.get("printedPower") or 0.0)
            features["card_hp"] = float(card_data.get("hp") or card_data.get("remainingHp") or card_data.get("currentHp") or 0.0)

        return features

    def _build_policy_observation(self, state: dict[str, Any], prompt_state: dict[str, Any]) -> list[float]:
        # Zero-pad the queue-mode feature vector to the policy's obs size (legacy
        # queue features occupy the first 30 slots; SWUEnv-trained checkpoints
        # expect the full 2386-dim tensor and act approximately randomly here).
        obs = [0.0] * int(getattr(self.policy, "obs_size", 64) or 64)
        phase = str(state.get("phase") or "unknown")
        prompt_type = str(prompt_state.get("promptType") or "unknown")
        prompt_title = str(prompt_state.get("menuTitle") or prompt_state.get("promptTitle") or "")

        for offset, values in ((0, _bucketed_hash_features(phase, 8)), (8, _bucketed_hash_features(prompt_type, 8)), (16, _bucketed_hash_features(prompt_title, 8))):
            for index, value in enumerate(values):
                obs[offset + index] = value

        players = state.get("players") or {}
        player_state = players.get(self.player_id) if isinstance(players, dict) else None
        if isinstance(player_state, dict):
            card_piles = player_state.get("cardPiles") or {}
            card_count = 0
            if isinstance(card_piles, dict):
                for pile_cards in card_piles.values():
                    if isinstance(pile_cards, list):
                        card_count += len(pile_cards)
            obs[24] = float(card_count)

        obs[25] = float(len(prompt_state.get("displayCards") or []))
        obs[26] = float(len(prompt_state.get("buttons") or []))
        obs[27] = float(len(prompt_state.get("perCardButtons") or []))
        obs[28] = float(len(prompt_state.get("dropdownListOptions") or []))
        obs[29] = 1.0 if prompt_state.get("distributeAmongTargets") else 0.0

        return obs

    def _choose_policy_candidate(self, state: dict[str, Any], prompt_state: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None

        if self.policy is None:
            return random.choice(candidates)

        if len(candidates) > int(getattr(self.policy, "max_actions", 0)):
            return random.choice(candidates)

        obs_tensor = torch.tensor(self._build_policy_observation(state, prompt_state), dtype=torch.float32, device=self.policy.device)

        with torch.no_grad():
            logits, _state_value = self.policy.masked_logits(obs_tensor, candidates)
            action_index = int(torch.argmax(logits).item())

        if 0 <= action_index < len(candidates):
            return candidates[action_index]

        return random.choice(candidates)

    def _find_card(self, state: dict[str, Any], card_uuid: Any) -> tuple[dict[str, Any], bool] | None:
        if not card_uuid:
            return None

        players = state.get("players") or {}
        if not isinstance(players, dict):
            return None

        for owner_id, player_state in players.items():
            if not isinstance(player_state, dict):
                continue

            if self._matches_card_uuid(player_state.get("leader"), card_uuid):
                return player_state.get("leader"), owner_id == self.player_id
            if self._matches_card_uuid(player_state.get("base"), card_uuid):
                return player_state.get("base"), owner_id == self.player_id

            card_piles = player_state.get("cardPiles") or {}
            if isinstance(card_piles, dict):
                for pile_cards in card_piles.values():
                    if not isinstance(pile_cards, list):
                        continue
                    for card in pile_cards:
                        if self._matches_card_uuid(card, card_uuid):
                            return card, owner_id == self.player_id

        return None

    @staticmethod
    def _matches_card_uuid(card: Any, card_uuid: Any) -> bool:
        return isinstance(card, dict) and str(card.get("uuid")) == str(card_uuid)

    def _find_done_button_for_current_prompt(self) -> dict[str, Any] | None:
        state = self.current_state or {}
        players = state.get("players") or {}
        if not isinstance(players, dict):
            return None

        player_state = players.get(self.player_id)
        if not isinstance(player_state, dict):
            return None

        prompt_state = player_state.get("promptState") or {}
        buttons = prompt_state.get("buttons") or []
        for button in buttons:
            if not isinstance(button, dict):
                continue
            text = str(button.get("text", "")).strip().lower()
            arg = str(button.get("arg", "")).strip().lower()
            if text == "done" or arg == "done":
                return button

        return None

    def _maybe_finish_from_game_state(self, game_state: dict[str, Any] | None) -> None:
        if self.game_over or not isinstance(game_state, dict):
            return

        winners = game_state.get("winnerNames") or game_state.get("winners")
        if isinstance(winners, list) and len(winners) > 0:
            self._finish_game(f"game state winners: {json.dumps(winners, default=str)}")
            return

        top_level_prompt = str(game_state.get("menuTitle") or game_state.get("promptTitle") or "").lower()
        if any(phrase in top_level_prompt for phrase in ("won the game", "the game ended in a draw", "game over")):
            self._finish_game(f"game state prompt: {top_level_prompt}")
            return

        players = game_state.get("players") or {}
        for player_state in players.values():
            if not isinstance(player_state, dict):
                continue

            prompt_state = player_state.get("promptState") or {}
            menu_title = str(prompt_state.get("menuTitle") or "").lower()
            if any(phrase in menu_title for phrase in ("won the game", "the game ended in a draw", "conceded", "game over")):
                self._finish_game(f"game state prompt: {menu_title}")
                return

    def _maybe_finish_from_lobby_state(self, lobby_state: dict[str, Any] | None) -> None:
        if self.game_over or not isinstance(lobby_state, dict):
            return

        game_ongoing = lobby_state.get("gameOngoing")
        win_history = lobby_state.get("winHistory") or {}
        if game_ongoing is False and self.seen_ongoing_game and (
            win_history.get("lastWinnerId") is not None
            or win_history.get("setEndResult") is not None
        ):
            self._finish_game(f"lobby state ended: {json.dumps(win_history, default=str)}")
            return

        if game_ongoing is False and self.seen_ongoing_game:
            self._finish_game("lobby state ended: gameOngoing=false")

    def _finish_game(self, reason: str) -> None:
        if self.game_over:
            return
        self.game_over = True
        self.game_over_reason = reason
        self._log(f"Game finished: {reason}")
        if self.socket.connected:
            self.socket.disconnect()

    def _collect_selectable_cards(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        players = state.get("players") or {}
        for player_state in players.values():
            if not isinstance(player_state, dict):
                continue

            for zone_cards in (player_state.get("cardPiles") or {}).values():
                if not isinstance(zone_cards, list):
                    continue
                for card in zone_cards:
                    if isinstance(card, dict) and card.get("selectable"):
                        cards.append(card)

            for key in ("leader", "base"):
                card = player_state.get(key)
                if isinstance(card, dict) and card.get("selectable"):
                    cards.append(card)

        return cards

    def _build_distribution_results(self, state: dict[str, Any], prompt_state: dict[str, Any]) -> dict[str, Any] | None:
        distribute_prompt = prompt_state.get("distributeAmongTargets") or {}
        amount = distribute_prompt.get("amount", 0)
        distribution_type = distribute_prompt.get("type")

        if amount <= 0 or not distribution_type:
            return None

        candidate_cards = self._distribution_candidates(state, prompt_state)
        if not candidate_cards:
            if distribute_prompt.get("canChooseNoTargets"):
                return {"type": distribution_type, "valueDistribution": []}
            return None

        max_targets = distribute_prompt.get("maxTargets") or 1
        chosen_cards = candidate_cards[:max_targets]
        if not chosen_cards:
            return None

        base_amount = amount // len(chosen_cards)
        remainder = amount % len(chosen_cards)
        value_distribution = []
        for index, card in enumerate(chosen_cards):
            card_amount = base_amount + (1 if index < remainder else 0)
            if card_amount > 0:
                value_distribution.append({"uuid": card.get("uuid"), "amount": card_amount})

        if not value_distribution and not distribute_prompt.get("canChooseNoTargets"):
            first_card = chosen_cards[0]
            value_distribution = [{"uuid": first_card.get("uuid"), "amount": amount}]

        return {
            "type": distribution_type,
            "valueDistribution": value_distribution,
        }

    def _distribution_candidates(self, state: dict[str, Any], prompt_state: dict[str, Any]) -> list[dict[str, Any]]:
        display_cards = prompt_state.get("displayCards") or []
        if display_cards:
            return [card for card in display_cards if isinstance(card, dict) and card.get("selectionState") != "invalid"]

        return self._collect_selectable_cards(state)

    def _emit_action(self, action: dict[str, Any]) -> None:
        action_type = action["actionType"]
        action_kind = action.get("kind", action_type)

        if action_kind == "statefulPromptResults":
            self.socket.emit("game", ("statefulPromptResults", action["result"], action["uuid"]))
            return

        if action_kind == "perCardMenuButton":
            self.socket.emit("game", ("perCardMenuButton", action.get("arg", ""), action.get("cardUuid"), action.get("uuid")))
            return

        if action_type == "macro_resource_cards":
            for card_uuid in action.get("uuids", []):
                self.socket.emit("game", ("cardClicked", card_uuid))
                time.sleep(0.05)

            done_button = self._find_done_button_for_current_prompt()
            if done_button is not None:
                self.socket.emit("game", ("menuButton", done_button.get("arg", "done"), done_button.get("uuid")))
            return

        if action_type == "clickPrompt":
            self.socket.emit("game", ("menuButton", action.get("arg", ""), action.get("uuid")))
            return

        if action_type == "clickCard":
            self.socket.emit("game", ("cardClicked", action.get("cardUuid")))
            return

        raise ValueError(f"Unsupported action type: {action_type}")


def main():
    parser = argparse.ArgumentParser(description="Single-seat SWU agent that can play against a GUI client or another bot.")
    parser.add_argument("-d", "--deck", required=True, help="Deck key to load from the decks file")
    parser.add_argument("--decks_file", type=str, default="decks.json", help="JSON file containing the deck dictionaries")
    parser.add_argument("--player_id", type=str, default="111", help="Server player id to control (111 or 222)")
    parser.add_argument("--server_url", type=str, help="Forceteki server URL. Defaults to the GUI server for queue mode and the RL server for --reset.")
    parser.add_argument("--policy_checkpoint", type=str, default="runs/train/policy_champion.ckpt", help="Checkpoint to use for policy decisions (champion ckpt or policy_latest.ckpt)")
    parser.add_argument("--reset", action="store_true", help="Initialize the game on the server before playing")
    parser.add_argument("--opponent_deck", type=str, help="Optional opponent deck key when using --reset")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps to take before stopping")
    parser.add_argument("--poll_delay", type=float, default=0.25, help="Seconds to wait between polls when it is not your turn")
    parser.add_argument("--verbose", action="store_true", help="Print board state and action details every turn")
    parser.add_argument("--debug_vis", action="store_true", help="Print the clean terminal board visualizer on every state update")
    args = parser.parse_args()

    server_url = args.server_url or ("http://localhost:3005" if args.reset else "http://localhost:9500")

    if not args.reset:
        # Human plays through the Forceteki GUI client (queue mode); the bot
        # takes the other seat automatically once matchmaking pairs them.
        print(f"Connecting to GUI queue server at {server_url} as player {args.player_id} using deck '{args.deck}'")
        bot = QueueBotClient(
            server_url=server_url,
            deck_key=args.deck,
            decks_file=args.decks_file,
            player_id=args.player_id,
            max_steps=args.max_steps,
            policy_checkpoint=args.policy_checkpoint,
            verbose=args.verbose,
            debug_vis=args.debug_vis,
        )
        try:
            bot.connect_and_queue()
            bot.wait()
            print(f"Done after {bot.steps_taken} steps.")
        except KeyboardInterrupt:
            print("Interrupted by user")
            if bot.socket.connected:
                bot.socket.disconnect()
        return

    env = SWUEnv(server_url=server_url, player_id=args.player_id)

    if args.reset:
        leader, base, deck = load_deck(args.deck, args.decks_file)
        payload = {
            "options": {
                "phase": "setup",
                "player1": {"hasInitiative": True},
            }
        }
        if args.player_id == "111":
            payload["p1Leader"] = leader
            payload["p1Base"] = base
            payload["p1Cards"] = deck
            if args.opponent_deck:
                opp = _load_optional_opponent_deck(args.opponent_deck, args.decks_file)
                payload["p2Leader"] = opp["leader"]
                payload["p2Base"] = opp["base"]
                payload["p2Cards"] = opp["cards"]
        else:
            payload["p2Leader"] = leader
            payload["p2Base"] = base
            payload["p2Cards"] = deck
            if args.opponent_deck:
                opp = _load_optional_opponent_deck(args.opponent_deck, args.decks_file)
                payload["p1Leader"] = opp["leader"]
                payload["p1Base"] = opp["base"]
                payload["p1Cards"] = opp["cards"]

        env.reset(options=payload)

    policy = None
    if args.policy_checkpoint and os.path.exists(_resolve_relative_path(args.policy_checkpoint)):
        torch_policy_module = importlib.import_module("torch_policy")
        policy_class = getattr(torch_policy_module, "TorchPolicy")
        policy = policy_class(
            obs_size=env.observation_space.shape[0],
            max_actions=env.action_space.n,
            device="cpu",
        )
        torch_module = importlib.import_module("torch")
        _load_policy_checkpoint(torch_module, policy, _resolve_relative_path(args.policy_checkpoint), "cpu")

    steps = 0
    terminated = False

    print(f"Connecting to {server_url} as player {args.player_id} using deck '{args.deck}'")
    print("Waiting for an existing game state on the server...")

    try:
        while not terminated and steps < args.max_steps:
            try:
                env.refresh()
            except Exception as exc:
                if not args.reset:
                    time.sleep(args.poll_delay)
                    continue
                raise RuntimeError(f"Unable to refresh server state: {exc}") from exc

            if args.debug_vis:
                render_rl_board(env)

            my_action_indices = [
                idx for idx, action in enumerate(env.available_actions)
                if action.get("playerId") == args.player_id
            ]

            if len(my_action_indices) == 0:
                time.sleep(args.poll_delay)
                continue

            if policy is not None:
                obs_tensor = torch.tensor(env._get_obs(), dtype=torch.float32)
                action_index, _log_prob, state_value = policy.select_action(
                    obs_tensor, env.available_actions, getattr(env, "legal_action_mask", None)
                )
                if action_index is None or action_index not in my_action_indices:
                    action_index = random.choice(my_action_indices)
                if args.debug_vis and state_value is not None:
                    print(f"[debug_vis] V(s) = {float(state_value):+.3f}")
            else:
                action_index = random.choice(my_action_indices)
            action = env.available_actions[action_index]
            print(f"[{args.player_id}] step {steps + 1}: action {action_index} -> {action.get('actionType')} {action.get('internalName', action.get('promptText', 'unknown'))}")

            _, reward, terminated, truncated, info = env.step(action_index)
            steps += 1

            print(f"[{args.player_id}] result: reward={reward}, terminated={terminated}, truncated={truncated}, phase={info.get('phase')}")

            if terminated or truncated:
                break

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Interrupted by user")

    print(f"Done after {steps} steps.")


if __name__ == "__main__":
    main()
