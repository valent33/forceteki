import copy
import itertools
import json
import os
import re
import warnings
from collections import Counter, deque
from typing import Any

import gymnasium as gym
import numpy as np
import requests
import sys as _sys
from gymnasium import spaces

from torch_policy import build_action_features

# ─────────────────────────────────────────────────────────────────────────────
# Star Wars: Unlimited domain dictionaries (Forceteki card data)
# ─────────────────────────────────────────────────────────────────────────────
ASPECTS = ["Aggression", "Command", "Cunning", "Vigilance", "Heroism", "Villainy"]

UNIT_TRAITS = [
    "Armor", "Bounty", "Bounty Hunter", "Capital Ship", "Clone", "Condition", "Creature",
    "Disaster", "Droid", "Ewok", "Fighter", "First Order", "Force", "Fortification", "Fringe",
    "Gambit", "Gungan", "Hutt", "Imperial", "Innate", "Inquisitor", "Item", "Jawa", "Jedi",
    "Kaminoan", "Law", "Learned", "Lightsaber", "Mandalorian", "Modification", "Musician",
    "Naboo", "New Republic", "Night", "Nihil", "Official", "Pilot", "Plan", "Rebel", "Republic",
    "Resistance", "Separatist", "Sith", "Spectre", "Speeder", "Supply", "Tactic", "Tank",
    "Transport", "Trick", "Trooper", "Tusken", "Twi'lek", "Undead", "Underworld", "Vehicle",
    "Walker", "Weapon", "Wookiee"
]

BASE_TRAITS = [
    "Aldhani", "Atollon", "Bracca", "Cantonica", "Castilon", "Christophsis", "Cloud City",
    "Concordia", "Corellia", "Coruscant", "Dagobah", "Dathomir", "Death Star", "D'Qar", "Eadu",
    "Endor", "Ferrix", "Geonosis", "Hosnian Prime", "Hoth", "Ilum", "Jedha", "Kalevala",
    "Kamino", "Kashyyyk", "Kessel", "Lothal", "Lowick", "Malachor", "Mandalore", "Mortis",
    "Mustafar", "Naboo", "Nadiri", "Narkina 5", "Nevarro", "Oba Diah", "Onderon", "Peridea",
    "Pillio", "Quarzite", "Ryloth", "Scarif", "Seatos", "Segra Milo", "Serenno", "Sorgan",
    "Starkiller Base", "Starlight Beacon", "Stygeon Prime", "Takodana", "Tatooine", "Utapau",
    "Vardos", "Vassek", "Wayland", "Yavin 4", "Zanbar", "Zeffo"
]

_ASPECT_INDEX: dict[str, int] = {aspect.lower(): i for i, aspect in enumerate(ASPECTS)}
_UNIT_TRAIT_INDEX: dict[str, int] = {trait.lower(): i for i, trait in enumerate(UNIT_TRAITS)}
_BASE_TRAIT_INDEX: dict[str, int] = {trait.lower(): i for i, trait in enumerate(BASE_TRAITS)}

# ─────────────────────────────────────────────────────────────────────────────
# State-tensor geometry (see `_get_obs()` for the exact per-float layout)
#   Block 1: global / force / economy                    → 14 floats
#   Block 2: bases & leaders (incl. 59-float multi-hot
#            BASE_TRAITS vector per base)                → 130 floats
#   Block 3: friendly hand (10 slots × 12 features)      → 120 floats
#   Block 4: opponent info-set densities (10 categories) → 10 floats
#   Block 5: arenas (ground & space, 6 friendly + 6 enemy
#            slots each, 88 features per slot)           → 2112 floats
#   TOTAL: OBS_DIM = 2386 floats
# ─────────────────────────────────────────────────────────────────────────────
HAND_SLOTS = 10
HAND_FEATURES = 12
ARENA_SLOTS_PER_SIDE = 6
UNIT_SLOT_FEATURES = 88

BLOCK1_SIZE = 14
BLOCK2_SIZE = 2 * (3 + len(BASE_TRAITS)) + 2 * 3
BLOCK3_SIZE = HAND_SLOTS * HAND_FEATURES
BLOCK4_SIZE = 10
BLOCK5_SIZE = 4 * ARENA_SLOTS_PER_SIDE * UNIT_SLOT_FEATURES

_OBS_B1 = 0
_OBS_B2 = _OBS_B1 + BLOCK1_SIZE
_OBS_B3 = _OBS_B2 + BLOCK2_SIZE
_OBS_B4 = _OBS_B3 + BLOCK3_SIZE
_OBS_B5 = _OBS_B4 + BLOCK4_SIZE
OBS_DIM = _OBS_B5 + BLOCK5_SIZE

# Category labels for Block 4 (opponent unseen-threat densities).
DECK_CATEGORY_LABELS = [
    "units", "events", "upgrades", "unique units", "space units",
    "ground units", "removal", "protection", "combat keywords", "cost >= 6",
]

_CARD_DB: dict[str, dict[str, Any]] | None = None


def _card_data_dir() -> str | None:
    """Locate the Forceteki card-data folder (test/json/Card)."""
    base = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(base, "..", "test", "json", "Card"),
        os.path.join(base, "..", "forceteki", "test", "json", "Card"),
        os.path.join(os.getcwd(), "test", "json", "Card"),
    ):
        try:
            if os.path.isdir(candidate):
                return os.path.normpath(candidate)
        except OSError:
            continue
    return None


def load_card_database(force: bool = False) -> dict[str, dict[str, Any]]:
    """Load the full card database keyed by `internalName` (cached after first use)."""
    global _CARD_DB
    if _CARD_DB is not None and not force:
        return _CARD_DB

    db: dict[str, dict[str, Any]] = {}
    card_dir = _card_data_dir()
    if card_dir:
        try:
            for fname in os.listdir(card_dir):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(card_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                except Exception:
                    continue
                entries = payload if isinstance(payload, list) else [payload]
                for entry in entries:
                    if isinstance(entry, dict) and entry.get("internalName"):
                        db[str(entry["internalName"])] = entry
        except OSError:
            pass
    _CARD_DB = db
    return db


class SWUEnv(gym.Env):
    """
    Custom Environment that connects headless via HTTP to the Forceteki Node.js server.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self, server_url="http://localhost:3005", player_id="111", single_agent_mode=False):
        super().__init__()
        self.server_url = server_url
        self.player_id = player_id
        self.single_agent_mode = single_agent_mode
        
        # Define maximum possible categorical actions
        # Actions in SWU arise from dynamically showing buttons + clicking cards.
        # Let's say action integer maps to the index of an available valid UI action.
        self.max_action_space = 100
        self.action_space = spaces.Discrete(self.max_action_space)

        # Some prompts list every card in the game (e.g. "Choose an option from
        # the list" → a dropdown with 1700+ card names). Cap dropdown fan-out so
        # the flat action list can never exceed the fixed policy-head size.
        self.max_dropdown_actions = 48

        # The observation is a structured, feature-complete State Tensor.
        # Exact layout is documented in `_get_obs()` and the module-level
        # block constants (OBS_DIM = 2386 floats).
        self.observation_space = spaces.Box(
            low=-10.0, high=100.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self.current_state = None
        self.available_actions = []
        self.active_players = []
        # Binary mask (length = action space) marking which available actions the
        # policy is allowed to choose. Built by `_update_available_actions()`.
        self.legal_action_mask = np.zeros(self.max_action_space, dtype=np.int8)
        # Track card UUIDs already clicked in multi-select prompts so the agent
        # can't keep picking the same cards — exhausts all options, then must click "Done".
        self._consumed_card_uuids: set[str] = set()
        self._last_prompt_key: str | None = None
        self._last_prompt_sig: str | None = None
        # No-progress loop breaker (see `_track_loop_repeat`): actions that get
        # retried from an identical (prompt, board) state are masked until the
        # prompt changes, so a saturated policy can't spin in play→Cancel cycles.
        self._loop_blocks: dict[str, tuple[str, str]] = {}
        self._step_history: deque[tuple] = deque(maxlen=8)
        self._loop_block_print_budget = 1
        # True when the most recent env.step() left the game state unchanged
        # (server did not accept the action) — consumed by the trainer to log
        # illegal actions with the server payload.
        self._last_step_no_progress = False
        # Deck definitions (internal names) captured from the /reset payload —
        # used to build the opponent info-set densities in Block 4.
        self._deck_definitions: dict[str, Counter] = {"player1": Counter(), "player2": Counter()}
        # UUID of the unit most recently selected as an attacker; consumed by
        # the arena / Sentinel legality rules in the action mask.
        self._pending_attacker_uuid: str | None = None
        # Most recently clicked card (used to promote a unit to attacker when
        # the player then presses the "Attack" button).
        self._last_clicked_card_uuid: str | None = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # POST /reset connects to Node.js backend
        payload = options if options else {
            "options": {
                "phase": "action",
                "player1": {"hasInitiative": True}
            }
        }

        # Remember both deck definitions (internal names) before the game starts.
        self._capture_deck_definitions(payload)

        resp = requests.post(f"{self.server_url}/reset", json=payload)
        resp.raise_for_status()

        self.current_state = resp.json()
        if "error" in self.current_state:
            raise RuntimeError(f"Error from server on reset: {self.current_state['error']}")

        # Fresh game → fresh loop-detection history.
        self._loop_blocks.clear()
        self._step_history.clear()
        self._loop_block_print_budget = 1
        self._update_available_actions()
        return self._get_obs(), self._get_info()

    def refresh(self):
        """Poll the shared server state without resetting the game."""
        resp = requests.get(f"{self.server_url}/state")
        resp.raise_for_status()

        self.current_state = resp.json()
        if "error" in self.current_state:
            raise RuntimeError(f"Error from server on refresh: {self.current_state['error']}")

        self._update_available_actions()
        return self._get_obs(), self._get_info()

    def _post_step(self, payload):
        """POST /step and return the resulting state dict.

        Contract assertions (HTTP 500, e.g. a stale card click whose prompt
        already resolved) must not kill the episode: on any transport/HTTP
        failure we resync from /state and return that instead. Returns None
        only if the server itself is unreachable."""
        try:
            resp = requests.post(f"{self.server_url}/step", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            try:
                state_resp = requests.get(f"{self.server_url}/state")
                state_resp.raise_for_status()
                return state_resp.json()
            except Exception:
                return None

    def _prompt_sig(self, state, player_id):
        """(promptUuid, menuTitle) of *player_id*'s prompt — used to detect
        prompt advancement between the posts of a multi-click macro."""
        seat = "player1" if str(state.get("player1Id")) == str(player_id) else "player2"
        prompt = (state.get("prompts") or {}).get(seat) or {}
        return (str(prompt.get("promptUuid")), str(prompt.get("menuTitle", "")))

    def _seat_for(self, state, player_id):
        return "player1" if str(state.get("player1Id")) == str(player_id) else "player2"

    def step(self, action_index):
        # Prevent invalid out-of-bounds actions
        if action_index >= len(self.available_actions):
            warnings.warn(f"Agent chose invalid action index {action_index}. Max is {len(self.available_actions)-1}.")
            # Penalize agent heavily for an illegal action and optionally terminate
            return self._get_obs(), -1.0, False, False, self._get_info()

        prev_state = copy.deepcopy(self.current_state)
        action_dict = self.available_actions[action_index]
        self._track_loop_repeat(action_dict)
        prev_progress_sig = self._state_progress_sig(prev_state)
        # Some prompts require structured results rather than a simple click.
        if action_dict.get("actionType") == "statefulPromptResults":
            payload = {
                "playerId": action_dict.get("playerId", self.player_id),
                "action": "statefulPromptResults",
                "arg": "any",
                "uuid": action_dict.get("uuid", ""),
                "method": "statefulPromptResults",
                "promptText": action_dict.get("promptText", ""),
                "result": action_dict.get("result"),
            }
            data = self._post_step(payload)
            if data is None:
                self.current_state = prev_state
            else:
                self.current_state = data
            reward = self._shape_reward(prev_state, self.current_state, action_dict)
            terminated = False
            truncated = False
            if "error" in self.current_state:
                print(f"Error on step: {self.current_state['error']}")
                terminated = True
                reward = -10.0
            else:
                winners = self.current_state.get("winners", [])
                if len(winners) > 0:
                    terminated = True
                elif self.current_state.get("phase") == "game_end":
                    terminated = True
                else:
                    prompts = self.current_state.get("prompts") or {}
                    for p in prompts.values():
                        if not p:
                            continue
                        title = str(p.get("menuTitle", "")).lower()
                        if "has won" in title or "has won the game" in title:
                            terminated = True
                            break
                    # Reliable fallback: base HP <= 0 means the game is over.
                    if not terminated:
                        state_section = self.current_state.get("state") or {}
                        for p_key in ("player1", "player2"):
                            player = state_section.get(p_key) or {}
                            base = player.get("base") or {}
                            hp = float(base.get("hp") or base.get("remainingHp") or base.get("currentHp") or 999)
                            if hp <= 0:
                                terminated = True
                                break

                self._update_available_actions()

            self._last_step_no_progress = prev_progress_sig == self._state_progress_sig(self.current_state)
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        if action_dict["actionType"] == "macro_resource_cards":
            player_id = str(action_dict.get("playerId", self.player_id))
            # 1. Click the cards. Stop early if the prompt advances or errors —
            #    leftover clicks would hit a different step and trip contract
            #    assertions on the server.
            for uid in action_dict.get("uuids", []):
                payload_card = {
                    "playerId": player_id,
                    "action": "clickCard",
                    "arg": "any",
                    "uuid": uid,
                    "method": "undefined",
                    "promptText": ""
                }
                data = self._post_step(payload_card)
                if data is None:
                    self.current_state = prev_state
                    break
                self.current_state = data
                if "error" in data:
                    break

            # 2. Resync, then click Done with fresh metadata from the CURRENT
            # prompt. Some prompts require specific arg/uuid/method values and
            # can stall otherwise.
            state = self.current_state
            if "error" not in state:
                try:
                    state_resp = requests.get(f"{self.server_url}/state")
                    state_resp.raise_for_status()
                    state = state_resp.json()
                    self.current_state = state
                except Exception:
                    pass

            p_key = self._seat_for(state, player_id)
            prompt = (state.get("prompts") or {}).get(p_key) or {}
            done_btn = next(
                (b for b in prompt.get("buttons", [])
                 if not b.get("disabled") and str(b.get("arg", "")).strip().lower() == "done"),
                None,
            )
            if done_btn is not None:
                payload_done = {
                    "playerId": player_id,
                    "action": "menuButton",
                    "arg": done_btn.get("arg", "done"),
                    "uuid": done_btn.get("uuid") or prompt.get("promptUuid", ""),
                    "method": done_btn.get("command", "menuButton"),
                    "promptText": done_btn.get("text", "Done")
                }
                data = self._post_step(payload_done)
                if data is not None:
                    self.current_state = data
        elif action_dict.get("actionType") == "perCardMenuButton":
            payload = {
                "playerId": action_dict.get("playerId", self.player_id),
                "action": "perCardMenuButton",
                "arg": action_dict.get("arg", "any"),
                "cardUuid": action_dict.get("cardUuid", ""),
                "uuid": action_dict.get("uuid", ""),
                "method": "menuButton",
                "promptText": "",
            }
            data = self._post_step(payload)
            if data is not None:
                self.current_state = data
        elif action_dict.get("actionType") == "displayCardClick":
            # DisplayCardsForSelectionPrompt (deck/trash/search) expects menuButton,
            # not cardClicked. Send a non-clickCard action so the server routes
            # through the fallback → game.menuButton(playerId, arg, ...).
            # The uuid MUST match the prompt's promptUuid (UiPrompt checks this).
            payload = {
                "playerId": action_dict.get("playerId", self.player_id),
                "action": "menuButton",
                "arg": action_dict.get("uuid", ""),
                "uuid": action_dict.get("promptUuid", ""),
                "method": "menuButton",
                "promptText": "",
            }
            data = self._post_step(payload)
            if data is not None:
                self.current_state = data
        elif action_dict.get("actionType") == "macro_select_all_cards":
            # Disclose-style prompt: click offered cards until the prompt
            # advances (or errors), then press Done from the CURRENT prompt.
            player_id = str(action_dict.get("playerId", self.player_id))
            before_sig = self._prompt_sig(self.current_state, player_id)
            clicked_any = False
            for uid in action_dict.get("uuids", []):
                payload_card = {
                    "playerId": player_id,
                    "action": "clickCard",
                    "arg": "any",
                    "uuid": uid,
                    "method": "undefined",
                    "promptText": "",
                }
                data = self._post_step(payload_card)
                if data is None:
                    break
                self.current_state = data
                clicked_any = True
                if "error" in data or self._prompt_sig(data, player_id) != before_sig:
                    # Single-select prompts auto-advance after the first click;
                    # multi-select prompts keep the same signature and need all.
                    break

            state = self.current_state
            if action_dict.get("doneArg") is not None:
                p_key = self._seat_for(state, player_id)
                prompt = (state.get("prompts") or {}).get(p_key) or {}
                done_btn = next(
                    (b for b in prompt.get("buttons", [])
                     if not b.get("disabled") and str(b.get("arg", "")).strip().lower() == "done"),
                    None,
                )
                if done_btn is not None:
                    payload_done = {
                        "playerId": player_id,
                        "action": "menuButton",
                        "arg": done_btn.get("arg", "done"),
                        "uuid": done_btn.get("uuid") or prompt.get("promptUuid", ""),
                        "method": done_btn.get("command", "menuButton"),
                        "promptText": done_btn.get("text", "Done"),
                    }
                    data = self._post_step(payload_done)
                    if data is not None:
                        self.current_state = data
            elif not clicked_any:
                # Nothing to click (e.g. the list was empty): just resync.
                try:
                    state_resp = requests.get(f"{self.server_url}/state")
                    state_resp.raise_for_status()
                    self.current_state = state_resp.json()
                except Exception:
                    pass
        else:
            # Generic actions (clickPrompt, clickCard, menuButton…).
            # Buttons are routed as `menuButton` with the button's arg + the
            # prompt's uuid — the engine dispatches by (uuid, arg), exactly how
            # the GUI client sends button presses. A stale click (prompt already
            # advanced) then becomes a harmless no-op instead of the server
            # raising TestSetupError on its text-based button lookup.
            action_type = action_dict.get("actionType", "menuButton")
            payload = {
                "playerId": action_dict.get("playerId", self.player_id),
                "action": "menuButton" if action_type in ("clickPrompt", "menuButton") else action_type,
                "arg": action_dict.get("arg", "any"),
                "uuid": action_dict.get("uuid", ""),
                "method": action_dict.get("method", "menuButton"),
                "promptText": action_dict.get("promptText", ""),
            }

            # Issue the action
            data = self._post_step(payload)
            if data is not None:
                self.current_state = data

        # Track consumed card UUID — prevents re-selection in multi-select prompts.
        if action_dict.get("actionType") in {"clickCard", "displayCardClick"} and action_dict.get("uuid"):
            self._consumed_card_uuids.add(str(action_dict["uuid"]))
        elif action_dict.get("actionType") == "perCardMenuButton" and action_dict.get("cardUuid"):
            self._consumed_card_uuids.add(str(action_dict["cardUuid"]))

        # Track the attacker for the follow-up "Choose a target for attack" prompt.
        clicked_uuid = action_dict.get("uuid")
        if action_dict.get("actionType") in {"clickCard", "displayCardClick"} and clicked_uuid:
            # Any clicked unit may later be promoted to attacker when the "Attack"
            # button is pressed, or when the previous prompt was an
            # "attack with ..." ability that selects the attacker itself.
            self._last_clicked_card_uuid = str(clicked_uuid)
            prev_prompts = (prev_state or {}).get("prompts") or {}
            for seat in ("player1", "player2"):
                prev_prompt = prev_prompts.get(seat) or {}
                title = str(prev_prompt.get("menuTitle", "")).lower()
                if "attack" in title and "with" in title:
                    self._pending_attacker_uuid = str(clicked_uuid)
                    break
        elif action_dict.get("actionType") == "clickPrompt":
            # Pressing an "Attack" button promotes the most recently clicked unit.
            action_text = f"{action_dict.get('promptText', '')} {action_dict.get('arg', '')}".lower()
            if "attack" in action_text and getattr(self, "_last_clicked_card_uuid", None):
                self._pending_attacker_uuid = self._last_clicked_card_uuid

        reward = self._shape_reward(prev_state, self.current_state, action_dict)
        terminated = False
        truncated = False

        if "error" in self.current_state:
            # Contract assertions on stale actions return {"error": ...}: the
            # game itself is still alive. Resync from /state and keep going
            # with a small penalty — only a dead server ends the episode.
            try:
                state_resp = requests.get(f"{self.server_url}/state")
                state_resp.raise_for_status()
                self.current_state = state_resp.json()
                terminated = False
                reward = -0.5
            except Exception:
                terminated = True
                reward = -10.0
        if not terminated:
            # check the winners field outputted by the server
            winners = self.current_state.get("winners", [])
            if len(winners) > 0:
                terminated = True
            elif self.current_state.get("phase") == "game_end":
                terminated = True
            else:
                # Some builds report the end-of-game via a prompt message.
                prompts = self.current_state.get("prompts") or {}
                for p in prompts.values():
                    if not p:
                        continue
                    title = str(p.get("menuTitle", "")).lower()
                    if "has won" in title or "has won the game" in title:
                        terminated = True
                        break

                # Reliable fallback: if either base is at 0 HP, the game is over.
                if not terminated:
                    state_section = self.current_state.get("state") or {}
                    for p_key in ("player1", "player2"):
                        player = state_section.get(p_key) or {}
                        base = player.get("base") or {}
                        hp = float(base.get("hp") or base.get("remainingHp") or base.get("currentHp") or 999)
                        if hp <= 0:
                            terminated = True
                            break

            self._update_available_actions()

        self._last_step_no_progress = prev_progress_sig == self._state_progress_sig(self.current_state)
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _update_available_actions(self):
        """
        Translates the current UI prompt / clickable states on the Node.js side
        into a flat list of valid actions, and constructs a binary
        `legal_action_mask` for dynamic action masking.

        The server is the single source of legality truth: `selectableCards` is
        engine-filtered (illegal targets such as Sentinel-guarded units are
        absent), buttons carry `disabled` flags, and `debug_legalActions`
        reports per-card requirement results. The env only builds actions from
        that data; the mask applies empirical no-progress loop blocks only.

        Returns
        -------
        numpy.ndarray
            Binary mask of length max_action_space (1 = legal). Also stored on
            `self.legal_action_mask`.
        """
        self.available_actions = []
        self.active_player = None
        self.active_players = []
        self.legal_action_mask = np.zeros(self.max_action_space, dtype=np.int8)

        if not self.current_state or "prompts" not in self.current_state:
            return self.legal_action_mask

        def _controlled_prompt_key() -> str | None:
            if not self.current_state:
                return None
            if str(self.current_state.get("player1Id")) == str(self.player_id):
                return "player1"
            if str(self.current_state.get("player2Id")) == str(self.player_id):
                return "player2"
            return None

        focus_key = None
        if self.single_agent_mode:
            def _has_interactive_display_cards(prompt: dict[str, Any] | None) -> bool:
                if not prompt:
                    return False

                if len(prompt.get("selectableCards", [])) > 0:
                    return True

                for card in prompt.get("displayCards", []) or []:
                    if not isinstance(card, dict):
                        continue
                    selection_state = str(card.get("selectionState", "")).lower()
                    if selection_state not in {"viewonly", "invalid", "unselectable", "selected"}:
                        return True

                return False

            def _is_stateful_prompt(prompt: dict[str, Any] | None) -> bool:
                """True if *prompt* has a recognised stateful type OR data key."""
                if not prompt:
                    return False
                ptype = prompt.get("promptType", "")
                if ptype in {"distributeAmongTargets", "chooseNumber", "chooseAmount", "number"}:
                    return True
                if any(k in prompt for k in ("chooseNumber", "chooseAmount", "selectNumber", "distributeAmongTargets")):
                    return True
                return False

            # In single-agent mode, prefer the prompt for the server-reported active player.
            active_player_id = str(self.current_state.get("activePlayer")) if self.current_state else None
            if active_player_id == str(self.current_state.get("player1Id")):
                candidate_key = "player1"
                candidate_prompt = self.current_state["prompts"].get(candidate_key) or {}
                title = str(candidate_prompt.get("menuTitle", ""))
                has_buttons = len(candidate_prompt.get("buttons", [])) > 0
                has_dropdowns = len(candidate_prompt.get("dropdownListOptions", [])) > 0
                has_cards = _has_interactive_display_cards(candidate_prompt)
                if title and "waiting for opponent" not in title.lower() and (has_buttons or has_dropdowns or has_cards or _is_stateful_prompt(candidate_prompt)):
                    focus_key = candidate_key
            elif active_player_id == str(self.current_state.get("player2Id")):
                candidate_key = "player2"
                candidate_prompt = self.current_state["prompts"].get(candidate_key) or {}
                title = str(candidate_prompt.get("menuTitle", ""))
                has_buttons = len(candidate_prompt.get("buttons", [])) > 0
                has_dropdowns = len(candidate_prompt.get("dropdownListOptions", [])) > 0
                has_cards = _has_interactive_display_cards(candidate_prompt)
                if title and "waiting for opponent" not in title.lower() and (has_buttons or has_dropdowns or has_cards or _is_stateful_prompt(candidate_prompt)):
                    focus_key = candidate_key

            # If the active player's prompt is not actionable yet, fall back to whichever
            # prompt is actually asking for a decision — even if it lacks buttons/cards
            # but has a recognised stateful prompt type or data key.
            for candidate_key in ("player1", "player2"):
                if focus_key is not None:
                    break
                candidate_prompt = self.current_state["prompts"].get(candidate_key) or {}
                title = str(candidate_prompt.get("menuTitle", ""))
                if title and "waiting for opponent" not in title.lower():
                    has_buttons = len(candidate_prompt.get("buttons", [])) > 0
                    has_dropdowns = len(candidate_prompt.get("dropdownListOptions", [])) > 0
                    has_cards = _has_interactive_display_cards(candidate_prompt)
                    if has_buttons or has_dropdowns or has_cards or _is_stateful_prompt(candidate_prompt):
                        focus_key = candidate_key
                        break

        if focus_key is None:
            focus_key = _controlled_prompt_key()

        if focus_key is None:
            # If we cannot map the env to a specific player, fall back to the active prompt.
            focus_key = "player1" if self.current_state.get("activePlayer") == self.current_state.get("player1Id") else "player2"

        p_key = focus_key
        p_id = self.current_state.get("player1Id") if p_key == "player1" else self.current_state.get("player2Id")
        player_prompt = self.current_state["prompts"].get(p_key)
        menu_title = (player_prompt or {}).get("menuTitle", "")

        # If the focused player is just waiting, see if the OTHER player has an
        # actionable prompt (e.g. opponent needs to choose a number mid‑turn).
        if player_prompt and "waiting for opponent" in menu_title.lower():
            other_key = "player2" if p_key == "player1" else "player1"
            other_prompt = self.current_state["prompts"].get(other_key)
            if other_prompt and "waiting for opponent" not in str(other_prompt.get("menuTitle", "")).lower():
                p_key = other_key
                p_id = self.current_state.get("player1Id") if p_key == "player1" else self.current_state.get("player2Id")
                player_prompt = other_prompt
                menu_title = (player_prompt or {}).get("menuTitle", "")

        # Detect prompt change — reset consumed card tracking when the prompt shifts.
        prompt_sig = f"{p_key}:{menu_title}:{player_prompt.get('promptUuid', '')}"
        if prompt_sig != getattr(self, "_last_prompt_sig", None):
            self._consumed_card_uuids.clear()
            self._last_prompt_sig = prompt_sig

        # The tracked attacker is only meaningful while an attack prompt is open.
        if "attack" not in str(menu_title or "").lower():
            self._pending_attacker_uuid = None

        if not player_prompt or "Waiting for opponent" in menu_title:
            return self.legal_action_mask

        # ── Number prompts are answered from the selectNumber data in the
        # last-resort fallback below (no debug print needed). ──

        has_buttons = "buttons" in player_prompt and len(player_prompt["buttons"]) > 0
        has_dropdowns = "dropdownListOptions" in player_prompt and len(player_prompt["dropdownListOptions"]) > 0
        selectable_uuids = player_prompt.get("selectableCards", [])
        display_cards = player_prompt.get("displayCards") or []

        is_stateful_distribution_prompt = (
            player_prompt and player_prompt.get("promptType") == "distributeAmongTargets" and player_prompt.get("distributeAmongTargets")
        )

        deferred_self_damage_buttons: list[dict[str, Any]] = []

        if has_buttons or has_dropdowns or len(selectable_uuids) > 0 or len(display_cards) > 0:
            self.active_player = p_id
            self.active_players.append(p_id)

            # Buttons that target yourself with damage are strictly worse than
            # targeting the opponent ("Deal 3 indirect damage to a player").
            # Build them but only offer them if the prompt has no other action.
            deferred_self_damage_buttons = []

            if has_buttons and not is_stateful_distribution_prompt:
                title_lower = str(menu_title or "").lower()
                for btn in player_prompt["buttons"]:
                    if btn.get("disabled", False):
                        continue
                    btn_arg = str(btn.get("arg", "")).strip().lower()
                    btn_text = str(btn.get("text", "")).strip().lower()
                    # Cancel is a zero-progress no-op: never offer it. If the
                    # agent picked a cancellable action it must commit to it
                    # (or pick something else) instead of undoing it.
                    if "cancel" in btn_arg or "cancel" in btn_text:
                        continue
                    # The server explicitly marks these as doing nothing.
                    if btn_text.startswith("(no effect)") or "(no effect)" in btn_text:
                        continue
                    # structured numeric features for the policy
                    features = {
                        "is_stateful": 0.0,
                        "is_macro": 0.0,
                        "is_dropdown": 1.0 if btn.get("command") == "menuButton" else 0.0,
                        "is_done": 1.0 if btn_arg == "done" else 0.0,
                        "is_claim": 1.0 if ("claim" in btn_arg or "claim" in btn_text) else 0.0,
                        "is_pass": 1.0 if ("pass" in btn_arg or "pass" in btn_text) else 0.0,
                        "is_card": 0.0,
                        "is_friendly": 0.0,
                        "is_leader": 0.0,
                        "is_base": 0.0,
                        "is_exhausted": 0.0,
                        "is_unit": 0.0,
                        "card_power": 0.0,
                        "card_hp": 0.0,
                        "is_self_target": 1.0 if "yourself" in btn_text else 0.0,
                        "is_opponent_target": 1.0 if "opponent" in btn_text else 0.0,
                    }

                    btn_action = {
                        "playerId": p_id,
                        "actionType": "clickPrompt",
                        "arg": btn.get("arg"),
                        "uuid": btn.get("uuid") or player_prompt.get("promptUuid", ""),
                        "method": btn.get("command"),
                        "promptText": btn.get("text"),
                        "features": features,
                    }
                    if "damage" in title_lower and "yourself" in btn_text:
                        deferred_self_damage_buttons.append(btn_action)
                        continue
                    self.available_actions.append(btn_action)

            if has_dropdowns:
                dropdown_options = list(player_prompt["dropdownListOptions"])
                # Reserve headroom for buttons / card actions and cap the fan-out.
                dropdown_cap = max(1, min(self.max_dropdown_actions, self.max_action_space - len(self.available_actions) - 4))
                dropdown_options = dropdown_options[:dropdown_cap]
                for option in dropdown_options:
                    features = {
                        "is_stateful": 0.0,
                        "is_macro": 0.0,
                        "is_dropdown": 1.0,
                        "is_done": 0.0,
                        "is_claim": 0.0,
                        "is_pass": 0.0,
                        "is_card": 0.0,
                        "is_friendly": 0.0,
                        "is_leader": 0.0,
                        "is_base": 0.0,
                        "is_exhausted": 0.0,
                        "is_unit": 0.0,
                        "card_power": 0.0,
                        "card_hp": 0.0,
                    }
                    self.available_actions.append({
                        "playerId": p_id,
                        "actionType": "clickPrompt",
                        "arg": option,
                        "uuid": player_prompt.get("promptUuid", ""),
                        "method": "menuButton",
                        "promptText": option,
                        "features": features,
                    })

            # Workaround for Node.js `ResourcePrompt.ts` clearing `selectableCards`
            # on the first tick before serialization.
            if "to resource" in menu_title and self.current_state.get("state") and p_key in self.current_state["state"]:
                my_state = self.current_state["state"][p_key]
                import itertools

                # Resourceing can only use the controlled player's hand.
                #   Setup:   "Select 2 cards to resource"           → every pair.
                #   Regroup: "Select between 0 and N cards to resource" → single
                #            cards (0 = the "Skip Resourcing" button, which the
                #            regular button loop above adds when enabled).
                hand_cards = [c for c in my_state.get("hand", []) if c.get("uuid")]
                if "between" in menu_title:
                    card_groups = [(card,) for card in hand_cards]
                elif "2 cards" in menu_title:
                    card_groups = list(itertools.combinations(hand_cards, 2))
                else:
                    card_groups = [(card,) for card in hand_cards]

                for group in card_groups:
                    if len(group) == 1:
                        card = group[0]
                        power_val = float(card.get("power") or card.get("printedPower") or 0.0)
                        hp_val = float(card.get("hp") or card.get("remainingHp") or card.get("currentHp") or 0.0)
                        features = {
                            "is_stateful": 0.0,
                            "is_macro": 1.0,
                            "is_dropdown": 0.0,
                            "is_done": 0.0,
                            "is_claim": 0.0,
                            "is_pass": 0.0,
                            "is_card": 1.0,
                            "is_friendly": 1.0,
                            "is_leader": 0.0,
                            "is_base": 0.0,
                            "is_exhausted": 0.0,
                            "is_unit": 0.0,
                            "card_power": power_val / 10.0,
                            "card_hp": hp_val / 20.0,
                        }
                        self.available_actions.append({
                            "playerId": p_id,
                            "actionType": "macro_resource_cards",
                            "uuids": [card["uuid"]],
                            "arg": "any",
                            "internalName": card.get("internalName", "Unknown"),
                            "cost": float(self._card_data(card.get("internalName")).get("cost") or 0.0),
                            "features": features,
                        })
                    else:
                        features = {
                            "is_stateful": 0.0,
                            "is_macro": 1.0,
                            "is_dropdown": 0.0,
                            "is_done": 0.0,
                            "is_claim": 0.0,
                            "is_pass": 0.0,
                            "is_card": 1.0,
                            "is_friendly": 1.0,
                            "is_leader": 0.0,
                            "is_base": 0.0,
                            "is_exhausted": 0.0,
                            "is_unit": 0.0,
                            "card_power": 0.0,
                            "card_hp": 0.0,
                        }
                        self.available_actions.append({
                            "playerId": p_id,
                            "actionType": "macro_resource_cards",
                            "uuids": [group[0]["uuid"], group[1]["uuid"]],
                            "arg": "any",
                            "internalName": f"{group[0].get('internalName', 'Unknown')} + {group[1].get('internalName', 'Unknown')}",
                            "cost": sum(float(self._card_data(card.get("internalName")).get("cost") or 0.0) for card in group),
                            "features": features,
                        })
            elif (len(selectable_uuids) > 0 or len(display_cards) > 0) and self.current_state.get("state") and p_key in self.current_state["state"] and not is_stateful_distribution_prompt:
                my_state = self.current_state["state"][p_key]

                seen_action_uuids: set[str] = set()
                        
                # Gather only cards that can legally be selected for this prompt.
                # Keep opponent zones only for prompts that can target enemy cards.
                # Track each card's owner ("me"/"enemy") and zone for the mask.
                all_cards: dict[str, dict[str, Any]] = {}
                card_owner: dict[str, str | None] = {}
                card_zone: dict[str, str | None] = {}
                for zone in ["hand", "spaceArena", "groundArena"]:
                    for card in my_state.get(zone, []):
                        all_cards[card["uuid"]] = card
                        card_owner[str(card["uuid"])] = "me"
                        card_zone[str(card["uuid"])] = self._zone_name(card)
                        for u in card.get("upgrades", []):
                            all_cards[u["uuid"]] = u
                            card_owner[str(u["uuid"])] = "me"
                            card_zone[str(u["uuid"])] = self._zone_name(card)
                if my_state.get("leader"):
                    all_cards[my_state["leader"]["uuid"]] = my_state["leader"]
                    card_owner[str(my_state["leader"]["uuid"])] = "me"
                    card_zone[str(my_state["leader"]["uuid"])] = self._zone_name(my_state["leader"])
                    for u in my_state["leader"].get("upgrades", []):
                        all_cards[u["uuid"]] = u
                        card_owner[str(u["uuid"])] = "me"
                        card_zone[str(u["uuid"])] = self._zone_name(my_state["leader"])
                if my_state.get("base"):
                    all_cards[my_state["base"]["uuid"]] = my_state["base"]
                    card_owner[str(my_state["base"]["uuid"])] = "me"
                    card_zone[str(my_state["base"]["uuid"])] = "base"
                    for u in my_state["base"].get("upgrades", []):
                        all_cards[u["uuid"]] = u
                        card_owner[str(u["uuid"])] = "me"
                        card_zone[str(u["uuid"])] = "base"

                opp_key = "player2" if p_key == "player1" else "player1"
                if opp_key in self.current_state["state"]:
                    opp_state = self.current_state["state"][opp_key]
                    for zone in ["spaceArena", "groundArena"]:
                        for card in opp_state.get(zone, []):
                            all_cards[card["uuid"]] = card
                            card_owner[str(card["uuid"])] = "enemy"
                            card_zone[str(card["uuid"])] = self._zone_name(card)
                            for u in card.get("upgrades", []):
                                all_cards[u["uuid"]] = u
                                card_owner[str(u["uuid"])] = "enemy"
                                card_zone[str(u["uuid"])] = self._zone_name(card)
                    if opp_state.get("leader"):
                        all_cards[opp_state["leader"]["uuid"]] = opp_state["leader"]
                        card_owner[str(opp_state["leader"]["uuid"])] = "enemy"
                        card_zone[str(opp_state["leader"]["uuid"])] = self._zone_name(opp_state["leader"])
                        for u in opp_state["leader"].get("upgrades", []):
                            all_cards[u["uuid"]] = u
                            card_owner[str(u["uuid"])] = "enemy"
                            card_zone[str(u["uuid"])] = self._zone_name(opp_state["leader"])
                    if opp_state.get("base"):
                        all_cards[opp_state["base"]["uuid"]] = opp_state["base"]
                        card_owner[str(opp_state["base"]["uuid"])] = "enemy"
                        card_zone[str(opp_state["base"]["uuid"])] = "base"
                        for u in opp_state["base"].get("upgrades", []):
                            all_cards[u["uuid"]] = u
                            card_owner[str(u["uuid"])] = "enemy"
                            card_zone[str(u["uuid"])] = "base"

                # Filter out cards already clicked in this multi-select prompt.
                filtered_selectable = [u for u in selectable_uuids if u not in self._consumed_card_uuids]
                for uuid, card in all_cards.items():
                    if uuid in filtered_selectable:
                        seen_action_uuids.add(str(uuid))
                        # is_friendly determined by whether the uuid appears in the controlled player's zones
                        is_friendly = False
                        try:
                            # check leader/base/hand/arenas for uuid membership
                            if my_state.get("leader") and my_state.get("leader").get("uuid") == uuid:
                                is_friendly = True
                            elif my_state.get("base") and my_state.get("base").get("uuid") == uuid:
                                is_friendly = True
                            else:
                                for zone in ("hand", "spaceArena", "groundArena"):
                                    if any(c.get("uuid") == uuid for c in my_state.get(zone, [])):
                                        is_friendly = True
                                        break
                        except Exception:
                            is_friendly = False

                        # detect leader/base/exhausted/unit
                        is_leader = False
                        is_base = False
                        try:
                            if my_state.get("leader") and my_state.get("leader").get("uuid") == uuid:
                                is_leader = True
                            if my_state.get("base") and my_state.get("base").get("uuid") == uuid:
                                is_base = True
                        except Exception:
                            pass

                        exhausted_flag = False
                        try:
                            exhausted_flag = bool(card.get("exhausted") or card.get("isExhausted") or card.get("is_exhausted"))
                        except Exception:
                            exhausted_flag = False

                        is_unit = True if (card.get("power") is not None or card.get("printedPower") is not None) else False
                        power_val = float(card.get("power") or card.get("printedPower") or 0.0)
                        hp_val = float(card.get("hp") or card.get("remainingHp") or card.get("currentHp") or 0.0)
                        features = {
                            "is_stateful": 0.0,
                            "is_macro": 0.0,
                            "is_dropdown": 0.0,
                            "is_done": 0.0,
                            "is_claim": 0.0,
                            "is_pass": 0.0,
                            "is_card": 1.0,
                            "is_friendly": 1.0 if is_friendly else 0.0,
                            "is_leader": 1.0 if is_leader else 0.0,
                            "is_base": 1.0 if is_base else 0.0,
                            "is_exhausted": 1.0 if exhausted_flag else 0.0,
                            "is_unit": 1.0 if is_unit else 0.0,
                            "card_power": power_val / 10.0,
                            "card_hp": hp_val / 20.0,
                        }

                        # Target legality metadata for the dynamic action mask.
                        target_owner = card_owner.get(str(uuid))
                        target_zone = card_zone.get(str(uuid))
                        target_kws = self._keyword_flags(self._card_data(card.get("internalName")))
                        meta = {
                            "targetUuid": str(uuid),
                            "targetOwner": target_owner,
                            "targetZone": target_zone,
                            "targetIsReady": not exhausted_flag,
                            "targetIsSentinel": bool(target_kws.get("sentinel")),
                            "targetKeywords": target_kws,
                        }

                        self.available_actions.append({
                            "playerId": p_id,
                            "actionType": "clickCard",
                            "uuid": card["uuid"],
                            "arg": "any",
                            "internalName": card.get("internalName", "Unknown"),
                            "features": features,
                            "meta": meta,
                        })

                # Some display-card prompts never populate selectableCards, but the visible cards are still clickable.
                # This is common for prompts that ask the player to order cards or choose from a hidden zone.
                for display_card in display_cards:
                    if not isinstance(display_card, dict):
                        continue

                    selection_state = str(display_card.get("selectionState", "")).lower()
                    # `unselectable` cards (e.g. L3-37 droids that would exceed
                    # the combined cost) are shown but NOT clickable — the
                    # server ignores clicks on them. Already-selected cards are
                    # excluded too so the policy can't toggle selections forever.
                    if selection_state in {"viewonly", "invalid", "unselectable", "selected"}:
                        continue

                    display_uuid = display_card.get("cardUuid") or display_card.get("uuid")
                    if not display_uuid or str(display_uuid) in seen_action_uuids:
                        continue

                    resolved_card = display_card
                    seen_action_uuids.add(str(display_uuid))

                    power_val = float(resolved_card.get("power") or resolved_card.get("printedPower") or 0.0)
                    hp_val = float(resolved_card.get("hp") or resolved_card.get("remainingHp") or resolved_card.get("currentHp") or 0.0)
                    features = {
                        "is_stateful": 0.0,
                        "is_macro": 0.0,
                        "is_dropdown": 0.0,
                        "is_done": 0.0,
                        "is_claim": 0.0,
                        "is_pass": 0.0,
                        "is_card": 1.0,
                        "is_friendly": 0.0,
                        "is_leader": 0.0,
                        "is_base": 0.0,
                        "is_exhausted": 1.0 if resolved_card.get("exhausted") or resolved_card.get("isExhausted") or resolved_card.get("is_exhausted") else 0.0,
                        "is_unit": 1.0 if (resolved_card.get("power") is not None or resolved_card.get("printedPower") is not None) else 0.0,
                        "card_power": power_val / 10.0,
                        "card_hp": hp_val / 20.0,
                    }

                    # Target legality metadata for the dynamic action mask.
                    d_owner, d_zone = self._locate_card_uuid(str(display_uuid), p_key)
                    d_kws = self._keyword_flags(
                        self._card_data(display_card.get("internalName") or resolved_card.get("internalName"))
                    )
                    d_exhausted = bool(
                        resolved_card.get("exhausted") or resolved_card.get("isExhausted")
                        or resolved_card.get("is_exhausted")
                    )
                    d_meta = {
                        "targetUuid": str(display_uuid),
                        "targetOwner": d_owner,
                        "targetZone": d_zone,
                        "targetIsReady": not d_exhausted,
                        "targetIsSentinel": bool(d_kws.get("sentinel")),
                        "targetKeywords": d_kws,
                    }

                    per_card_buttons = player_prompt.get("perCardButtons") or []
                    if per_card_buttons:
                        # DisplayCardsWithButtonsPrompt: each card has buttons (e.g. "Top"/"Bottom").
                        # Create a separate action for every (card, button) pair.
                        for btn in per_card_buttons:
                            btn_features = {**features}
                            btn_features["is_done"] = 1.0 if str(btn.get("arg", "")).lower() == "done" else 0.0
                            btn_action_label = btn.get("text", str(btn.get("arg", "")))
                            card_name = display_card.get("internalName", resolved_card.get("internalName", "Unknown"))
                            self.available_actions.append({
                                "playerId": p_id,
                                "actionType": "perCardMenuButton",
                                "cardUuid": str(display_uuid),
                                "arg": btn.get("arg", "any"),
                                "uuid": player_prompt.get("promptUuid", ""),
                                "internalName": f"{card_name} → {btn_action_label}",
                                "features": btn_features,
                                "meta": d_meta,
                            })
                    else:
                        self.available_actions.append({
                            "playerId": p_id,
                            "actionType": "displayCardClick",
                            "uuid": str(display_uuid),
                            "arg": str(display_uuid),
                            "promptUuid": player_prompt.get("promptUuid", ""),
                            "internalName": display_card.get("internalName", resolved_card.get("internalName", "Unknown")),
                            "features": features,
                            "meta": d_meta,
                        })

        # Structured distribution prompts need a single synthetic action that carries the server result.
        if is_stateful_distribution_prompt:
            distribution = self._build_distribution_results_for_prompt(p_key)
            if distribution is not None:
                self.available_actions.append({
                    "playerId": p_id,
                    "actionType": "statefulPromptResults",
                    "uuid": player_prompt.get("promptUuid", ""),
                    "result": distribution,
                    "promptText": player_prompt.get("menuTitle", ""),
                    "internalName": f"statefulPromptResults {distribution.get('type')}",
                })

        # ── Last‑resort fallback: no actions built for an active prompt ──
        if len(self.available_actions) == 0 and player_prompt and "waiting for opponent" not in menu_title.lower():
            prompt_type = player_prompt.get("promptType", "")
            menu_lower = menu_title.lower()
            is_number_prompt = ("choose a number" in menu_lower or
                                "choose number" in menu_lower or
                                prompt_type in ("chooseNumber", "chooseAmount", "number") or
                                any(k in player_prompt for k in ("chooseNumber", "chooseAmount", "selectNumber")))

            # ── ALWAYS dump on number prompts so we can debug ──────
            if is_number_prompt:
                try:
                    import json as _json, os as _os
                    dump_path = _os.path.join(_os.getcwd(), "prompt_debug.json")
                    with open(dump_path, "w") as _f:
                        _json.dump({
                            "p_key": p_key,
                            "p_id": p_id,
                            "menu_title": menu_title,
                            "prompt_keys": sorted(player_prompt.keys()),
                            "prompt_type": prompt_type,
                            "buttons": [{"text": b.get("text",""), "arg": b.get("arg",""),
                                         "disabled": b.get("disabled", False),
                                         "command": b.get("command","")}
                                        for b in player_prompt.get("buttons", [])],
                            "dropdown_count": len(player_prompt.get("dropdownListOptions", [])),
                            "selectableCards_count": len(player_prompt.get("selectableCards", [])),
                            "displayCards_count": len(player_prompt.get("displayCards") or []),
                            "selectNumber_data": str(player_prompt.get("selectNumber", "MISSING"))[:500],
                            "prompt_title": str(player_prompt.get("promptTitle", "MISSING"))[:200],
                            "chooseNumber_data": str(player_prompt.get("chooseNumber", "MISSING"))[:500],
                            "chooseAmount_data": str(player_prompt.get("chooseAmount", "MISSING"))[:500],
                        }, _f, indent=2, default=str)
                except Exception:
                    pass

            # ── Try every known way to answer a NumberPrompt ────────
            # NumberPrompt is answered with a menuButton whose `arg` is the
            # chosen integer (server asserts min <= arg <= max). It does NOT
            # support statefulPromptResults — sending that crashes the server
            # contract ("Attempting to trigger onStatefulPromptResults ... not
            # supported by the current step").
            choose_key = None
            for k in ("selectNumber", "chooseNumber", "chooseAmount"):
                if k in player_prompt:
                    choose_key = k
                    break
            if not choose_key and prompt_type in ("chooseNumber", "chooseAmount", "number"):
                choose_key = "selectNumber"
            if choose_key:
                choose_data = player_prompt.get(choose_key) or {}
                if not isinstance(choose_data, dict):
                    choose_data = {}
                minimum = choose_data.get("min", choose_data.get("minimum"))
                maximum = choose_data.get("max", choose_data.get("maximum"))
                default = choose_data.get("value", choose_data.get("amount"))
                try:
                    lo = int(minimum)
                    hi = int(maximum)
                except (TypeError, ValueError):
                    lo, hi = None, None
                if lo is not None and hi is not None and hi >= lo:
                    # Offer the full choice range (capped at 12) so the policy
                    # can pick meaningfully instead of always answering the
                    # default/min value.
                    values = list(range(lo, hi + 1))[:12]
                    for val in values:
                        self.available_actions.append({
                            "playerId": p_id,
                            # `action: menuButton` is routed by envServer.ts straight to
                            # game.menuButton(arg, uuid). NumberPrompt serializes NO buttons,
                            # so a `clickPrompt` would fail the button lookup and return 500.
                            "actionType": "menuButton",
                            "arg": str(val),
                            "uuid": player_prompt.get("promptUuid", ""),
                            "method": "menuButton",
                            "promptText": f"{menu_title}: {val}",
                            "internalName": f"numberPrompt {val}",
                            "features": {
                                "is_stateful": 1.0,
                                "is_macro": 0.0,
                                "is_dropdown": 0.0,
                                "is_done": 0.0,
                                "is_claim": 0.0,
                                "is_pass": 0.0,
                                "is_card": 0.0,
                                "is_friendly": 0.0,
                                "is_leader": 0.0,
                                "is_base": 0.0,
                                "is_exhausted": 0.0,
                                "is_unit": 0.0,
                                "card_power": 0.0,
                                "card_hp": 0.0,
                            },
                        })

            # 2) Any non‑disabled button (many NumberPrompts have +/-/Done)
            if len(self.available_actions) == 0:
                for btn in player_prompt.get("buttons", []):
                    if not btn.get("disabled", False):
                        if "cancel" in str(btn.get("arg", "")).lower() or "cancel" in str(btn.get("text", "")).lower():
                            continue
                        self.available_actions.append({
                            "playerId": p_id,
                            "actionType": "clickPrompt",
                            "arg": btn.get("arg", "done"),
                            "uuid": btn.get("uuid", player_prompt.get("promptUuid", "")),
                            "method": btn.get("command", "menuButton"),
                            "promptText": btn.get("text", "Continue"),
                        })
                        break

            # 3) Desperate: any remaining non-disabled button.
            #    (Never click DISABLED buttons — that just loops forever, e.g. the
            #    setup "Confirm Resources" button before 2 cards are selected.)
            if len(self.available_actions) == 0:
                for btn in player_prompt.get("buttons", []):
                    if btn.get("disabled", False):
                        continue
                    if "cancel" in str(btn.get("arg", "")).lower() or "cancel" in str(btn.get("text", "")).lower():
                        continue
                    self.available_actions.append({
                        "playerId": p_id,
                        "actionType": "clickPrompt",
                        "arg": btn.get("arg", "done"),
                        "uuid": btn.get("uuid", player_prompt.get("promptUuid", "")),
                        "method": btn.get("command", "menuButton"),
                        "promptText": btn.get("text", "Continue"),
                    })
                    break

            # 4) Absolute last resort: deliberately build NOTHING. Inventing a
            #    raw "Done" menuButton for an unknown prompt hits the server's
            #    contract assertion ("...not supported by the current step") and
            #    corrupts the game. Zero actions lets the caller poll and abort
            #    the episode via the stall guard instead of crashing the server.

        # ── Disclose prompts: the server lists every card that may be disclosed.
        # When the disclosure is optional ("you may choose nothing"), the
        # 'Choose nothing' button is always legal and avoids multi-click races;
        # prefer it. Otherwise collapse the selection into one macro action so
        # the policy doesn't have to click cards one by one.
        if "disclose" in str(menu_title or "").lower():
            buttons_now = player_prompt.get("buttons") or []
            choose_nothing = next(
                (b for b in buttons_now
                 if not b.get("disabled") and "choose nothing" in str(b.get("text", "")).strip().lower()),
                None,
            )
            if choose_nothing is not None:
                self.available_actions = [{
                    "playerId": p_id,
                    "actionType": "clickPrompt",
                    "arg": choose_nothing.get("arg", "done"),
                    "uuid": choose_nothing.get("uuid") or player_prompt.get("promptUuid", ""),
                    "method": choose_nothing.get("command", "menuButton"),
                    "promptText": choose_nothing.get("text", "Choose nothing"),
                    "internalName": "disclose choose nothing",
                    "features": {
                        "is_stateful": 0.0,
                        "is_macro": 0.0,
                        "is_dropdown": 0.0,
                        "is_done": 1.0,
                        "is_claim": 0.0,
                        "is_pass": 0.0,
                        "is_card": 0.0,
                        "is_friendly": 0.0,
                        "is_leader": 0.0,
                        "is_base": 0.0,
                        "is_exhausted": 0.0,
                        "is_unit": 0.0,
                        "card_power": 0.0,
                        "card_hp": 0.0,
                    },
                }]
            else:
                disclose_uuids = [u for u in (selectable_uuids or []) if u not in self._consumed_card_uuids]
                done_btn = next(
                    (b for b in buttons_now
                     if not b.get("disabled") and str(b.get("arg", "")).strip().lower() == "done"),
                    None,
                )
                if disclose_uuids or done_btn is not None:
                    self.available_actions = [{
                        "playerId": p_id,
                        "actionType": "macro_select_all_cards",
                        "uuids": list(disclose_uuids),
                        "doneArg": done_btn.get("arg") if done_btn else None,
                        "doneText": done_btn.get("text") if done_btn else None,
                        "uuid": player_prompt.get("promptUuid", ""),
                        "internalName": f"disclose {len(disclose_uuids)} card(s)",
                        "features": {
                            "is_stateful": 1.0,
                            "is_macro": 1.0,
                            "is_dropdown": 0.0,
                            "is_done": 0.0,
                            "is_claim": 0.0,
                            "is_pass": 0.0,
                            "is_card": 0.0,
                            "is_friendly": 0.0,
                            "is_leader": 0.0,
                            "is_base": 0.0,
                            "is_exhausted": 0.0,
                            "is_unit": 0.0,
                            "card_power": 0.0,
                            "card_hp": 0.0,
                        },
                    }]

        # ── Attach per-action feature vectors for the Q(s,a) policy head ─────
        # Strict-dominance fallback: self-damage buttons are only offered when
        # the prompt has no other action at all.
        if not self.available_actions and deferred_self_damage_buttons:
            self.available_actions = list(deferred_self_damage_buttons)

        ready_resources = 0.0
        state_section = self.current_state.get("state") or {}
        if state_section and p_key in state_section:
            ready_resources = self._num(state_section[p_key], "readyResourceCount")
        for action in self.available_actions:
            cost = 0.0
            if action.get("actionType") == "clickCard" and action.get("internalName"):
                cost = float(self._card_data(action.get("internalName")).get("cost") or 0.0)
                action["playable"] = 1.0 if cost <= ready_resources + 1e-6 else 0.0
            else:
                action["playable"] = 1.0
            action["action_features"] = build_action_features(action, cost=cost)

        # ── Hard safety cap ──────────────────────────────────────────────────
        # The policy head has exactly `max_action_space` slots, so the flat
        # action list (and the mask built from it) must never exceed that.
        if len(self.available_actions) > self.max_action_space:
            self.available_actions = self.available_actions[:self.max_action_space]

        # ── Dynamic action masking: mark which of these actions the policy may take ──
        self._apply_dynamic_action_masking(player_prompt, p_key)
        return self.legal_action_mask


    # ──────────────────────────────────────────────────────────────────────────
    # Card database / domain helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _card_data(self, internal_name: Any) -> dict[str, Any]:
        name = str(internal_name or "").strip().lower()
        if not name:
            return {}
        return load_card_database().get(name, {})

    def _capture_deck_definitions(self, payload: Any) -> None:
        """Record both deck definitions (internal names) from the /reset payload."""
        self._deck_definitions = {"player1": Counter(), "player2": Counter()}
        if not isinstance(payload, dict):
            return
        for seat_key, field in (("player1", "p1Cards"), ("player2", "p2Cards")):
            cards = payload.get(field)
            if isinstance(cards, (list, tuple)):
                self._deck_definitions[seat_key] = Counter(str(card) for card in cards)
            elif isinstance(cards, dict):
                for raw_id, count in cards.items():
                    try:
                        self._deck_definitions[seat_key][str(raw_id)] += int(count)
                    except (TypeError, ValueError):
                        continue

    def _my_opp_keys(self, state: dict[str, Any] | None) -> tuple[str, str]:
        if not state:
            return "player1", "player2"
        if str(state.get("player1Id")) == str(self.player_id):
            return "player1", "player2"
        if str(state.get("player2Id")) == str(self.player_id):
            return "player2", "player1"
        return "player1", "player2"

    def _num(self, obj: Any, *keys: str, default: float = 0.0) -> float:
        if not isinstance(obj, dict):
            return float(default)
        for key in keys:
            value = obj.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)

    def _is_exhausted(self, card: Any) -> bool:
        if not isinstance(card, dict):
            return False
        return bool(card.get("exhausted") or card.get("isExhausted") or card.get("is_exhausted"))

    def _max_hp(self, state_card: Any, db_card: dict[str, Any]) -> float:
        if db_card.get("hp") is not None:
            try:
                return float(db_card["hp"])
            except (TypeError, ValueError):
                pass
        hp = self._num(state_card, "hp", "currentHp")
        damage = self._num(state_card, "damage")
        return hp + damage

    def _current_hp(self, state_card: Any, db_card: dict[str, Any]) -> float:
        hp = self._num(state_card, "hp", "currentHp")
        if hp > 0.0 or not isinstance(state_card, dict):
            return hp
        return max(0.0, self._max_hp(state_card, db_card) - self._num(state_card, "damage"))

    def _zone_name(self, state_card: Any) -> str | None:
        if not isinstance(state_card, dict):
            return None
        zone = str(state_card.get("zone") or "").lower()
        if "ground" in zone:
            return "ground"
        if "space" in zone:
            return "space"
        if "hand" in zone:
            return "hand"
        if "deck" in zone:
            return "deck"
        if "discard" in zone:
            return "discard"
        if "resource" in zone:
            return "resource"
        if "leader" in zone:
            return "leader"
        if "base" in zone:
            return "base"
        arena = str(self._card_data(state_card.get("internalName")).get("arena") or "").lower()
        if arena in ("ground", "space"):
            return arena
        return None

    def _aspect_hot(self, aspects: Any) -> np.ndarray:
        hot = np.zeros(len(ASPECTS), dtype=np.float32)
        for aspect in aspects or []:
            index = _ASPECT_INDEX.get(str(aspect).lower())
            if index is not None:
                hot[index] = 1.0
        return hot

    def _trait_hot(self, traits: Any, index_map: dict[str, int], size: int) -> np.ndarray:
        hot = np.zeros(size, dtype=np.float32)
        for trait in traits or []:
            index = index_map.get(str(trait).lower())
            if index is not None:
                hot[index] = 1.0
        return hot

    def _keyword_flags(self, db_card: dict[str, Any]) -> dict[str, Any]:
        """Combat-keyword flags plus Raid/Restore values parsed from card text."""
        keywords = {str(k).lower() for k in (db_card.get("keywords") or [])}
        text = f"{db_card.get('text') or ''} {db_card.get('deployBox') or ''}".lower()
        flags: dict[str, Any] = {
            "sentinel": "sentinel" in keywords,
            "saboteur": "saboteur" in keywords,
            "grit": "grit" in keywords,
            "overwhelm": "overwhelm" in keywords,
            "ambush": "ambush" in keywords,
            "hidden": ("hidden" in keywords) or ("stealth" in keywords),
            "raid": 0.0,
            "restore": 0.0,
        }
        match = re.search(r"raid\s*(\d+)", text)
        flags["raid"] = float(int(match.group(1))) if match else (1.0 if "raid" in keywords else 0.0)
        match = re.search(r"restore\s*(\d+)", text)
        flags["restore"] = float(int(match.group(1))) if match else (1.0 if "restore" in keywords else 0.0)
        return flags

    def _card_type_scalar(self, db_card: dict[str, Any]) -> float:
        types = {str(t).lower() for t in (db_card.get("types") or [])}
        if "unit" in types:
            return 1.0
        if "event" in types:
            return 2.0
        if "upgrade" in types:
            return 3.0
        return 0.0

    def _locate_card_uuid(self, uuid: str, my_key: str) -> tuple[str | None, str | None]:
        """Return (owner, zone) for a card uuid on the current board, if present."""
        state_section = (self.current_state or {}).get("state") or {}
        opp_key = "player2" if my_key == "player1" else "player1"
        for owner, seat in (("me", my_key), ("enemy", opp_key)):
            player_state = state_section.get(seat) or {}
            for zone_key in ("hand", "spaceArena", "groundArena"):
                for card in player_state.get(zone_key) or []:
                    if str(card.get("uuid")) == uuid:
                        return owner, self._zone_name(card)
                    for upgrade in card.get("upgrades") or []:
                        if str(upgrade.get("uuid")) == uuid:
                            return owner, self._zone_name(card)
            for special in ("leader", "base"):
                card = player_state.get(special)
                if isinstance(card, dict) and str(card.get("uuid")) == uuid:
                    return owner, self._zone_name(card)
                for upgrade in (card or {}).get("upgrades") or []:
                    if str(upgrade.get("uuid")) == uuid:
                        return owner, self._zone_name(card)
        return None, None

    def _debug_playable_map(self, state: dict[str, Any]) -> dict[str, bool]:
        """Playable status per hand card from the server's debug_legalActions."""
        playable: dict[str, bool] = {}
        for seat in ("player1", "player2"):
            prompt = (state.get("prompts") or {}).get(seat) or {}
            for entry in prompt.get("debug_legalActions") or []:
                if not isinstance(entry, dict):
                    continue
                can_play = any(
                    bool(action.get("req")) and bool(action.get("isPlay"))
                    for action in entry.get("actions") or [] if isinstance(action, dict)
                )
                playable[str(entry.get("id") or "").strip().lower()] = can_play
        return playable

    def _classify_deck_card(self, db_card: dict[str, Any]) -> tuple[int, ...]:
        """Map a card definition to Block-4 threat category indices."""
        if not db_card:
            return ()
        types = {str(t).lower() for t in (db_card.get("types") or [])}
        text = str(db_card.get("text") or "").lower()
        keywords = {str(k).lower() for k in (db_card.get("keywords") or [])}
        arena = str(db_card.get("arena") or "").lower()
        categories: list[int] = []
        if "unit" in types:
            categories.append(0)
        if "event" in types:
            categories.append(1)
        if "upgrade" in types:
            categories.append(2)
        if "unit" in types and db_card.get("unique"):
            categories.append(3)
        if "unit" in types and arena == "space":
            categories.append(4)
        if "unit" in types and arena == "ground":
            categories.append(5)
        if re.search(r"damage|defeat|destroy", text):
            categories.append(6)
        if re.search(r"shield|restore|heal", text):
            categories.append(7)
        if keywords & {"ambush", "sentinel", "saboteur", "grit", "overwhelm", "raid"}:
            categories.append(8)
        if (db_card.get("cost") or 0) >= 6:
            categories.append(9)
        return tuple(categories)

    def _opponent_deck_densities(self, opp_key: str, opp_state: dict[str, Any]) -> np.ndarray:
        """Block 4: (deck definition - discard pile) / remaining deck, per category."""
        densities = np.zeros(BLOCK4_SIZE, dtype=np.float32)
        definition = self._deck_definitions.get(opp_key) or Counter()
        remaining = max(1, len(opp_state.get("deck") or []))
        def_count = [0.0] * BLOCK4_SIZE
        discard_count = [0.0] * BLOCK4_SIZE
        if definition:
            for name, count in definition.items():
                for category in self._classify_deck_card(self._card_data(name)):
                    def_count[category] += float(count)
        for card in opp_state.get("discard") or []:
            for category in self._classify_deck_card(self._card_data(card.get("internalName"))):
                discard_count[category] += 1.0
        for category in range(BLOCK4_SIZE):
            if definition:
                density = max(0.0, def_count[category] - discard_count[category]) / float(remaining)
            else:
                # No deck definition available: fall back to discard-pile composition.
                density = discard_count[category] / float(remaining)
            densities[category] = min(1.0, density)
        return densities

    # ──────────────────────────────────────────────────────────────────────────
    # Dynamic action-masking helpers
    # ──────────────────────────────────────────────────────────────────────────
    CARD_ACTION_TYPES = ("clickCard", "displayCardClick", "perCardMenuButton", "macro_resource_cards")

    def _apply_dynamic_action_masking(self, player_prompt: dict[str, Any] | None, p_key: str) -> np.ndarray:
        """Build the binary legal_action_mask for the current action list.

        The server is the single source of legality truth and its data is
        consumed at build time: `selectableCards` is engine-filtered (verified
        against recorded games — illegal attack targets such as Sentinel-guarded
        units are absent from the list), and disabled buttons are skipped.
        This mask therefore reimplements NO game rules; it only applies the
        empirical no-progress loop blocks from `_track_loop_repeat`.
        """
        actions = self.available_actions[:self.max_action_space]
        n = len(actions)
        mask = np.zeros(self.max_action_space, dtype=np.int8)
        if n == 0:
            self.legal_action_mask = mask
            return mask

        # Actions blocked by no-progress loop detection while this exact prompt
        # instance is open (see `_track_loop_repeat`).
        current_prompt_sig = (
            p_key,
            str((player_prompt or {}).get("menuTitle", "")),
            str((player_prompt or {}).get("promptUuid", "")),
        )
        blocked_keys = {key for key, sig in self._loop_blocks.items() if sig == current_prompt_sig}

        for index, action in enumerate(actions):
            if self._action_key(action) in blocked_keys:
                continue
            mask[index] = 1

        # Safety valve: never leave an active prompt with zero legal actions.
        if int(mask.sum()) == 0:
            mask[:n] = 1
        self.legal_action_mask = mask
        return mask

    # ── No-progress loop breaker ─────────────────────────────────────────────
    @staticmethod
    def _action_key(action: dict[str, Any]) -> str:
        """Compact identity of an action (deliberately ignores prompt uuids,
        which the server regenerates on every prompt re-issue)."""
        action_type = str(action.get("actionType") or "")
        if action_type == "clickPrompt":
            return f"clickPrompt:{action.get('arg')}:{action.get('promptText')}"
        if action_type == "clickCard":
            return f"clickCard:{action.get('uuid')}"
        if action_type == "perCardMenuButton":
            # Include the target card: ordering prompts (e.g. Qui-Gon's "Put on
            # top") share one arg for every card, and clicking a DIFFERENT card
            # is legitimate progress, not a loop.
            return f"perCardMenuButton:{action.get('arg')}:{action.get('cardUuid')}"
        if action_type == "menuButton":
            return f"menuButton:{action.get('arg')}:{action.get('promptText')}"
        if action_type == "macro_resource_cards":
            return f"macro:{tuple(action.get('uuids') or ())}"
        return f"{action_type}:{action.get('arg')}:{action.get('uuid')}"

    def _player_state_sig(self, state: dict[str, Any] | None, p_id: str) -> str:
        """Compact signature of BOTH players' boards (units, bases, economy).

        Includes per-unit exhaustion and damage so that legal progress — e.g.
        attacking the same target twice — changes the signature and is not
        mistaken for a no-progress loop."""
        if not state:
            return ""
        parts: list[str] = []
        for p_key in ("player1", "player2"):
            sec = (state.get("state") or {}).get(p_key) or {}
            hand = tuple(sorted(str(c.get("uuid", "")) for c in sec.get("hand", []) if c.get("uuid")))
            zones = tuple(sorted(
                (
                    str(c.get("uuid", "")),
                    1 if (c.get("exhausted") or c.get("isExhausted")) else 0,
                    int(c.get("damage") or 0),
                )
                for zone in ("spaceArena", "groundArena")
                for c in sec.get(zone, []) if c.get("uuid")
            ))
            base = sec.get("base") or {}
            leader = sec.get("leader") or {}
            parts.append(f"{hand}|{zones}|{sec.get('credits')}|{sec.get('readyResources')}|{base.get('hp')}|{leader.get('hp')}")
        return " || ".join(parts)

    def _state_progress_sig(self, state: dict[str, Any] | None) -> str:
        """Signature of prompts + board, used to detect steps the server did
        not accept (identical before/after state). Includes buttons and
        display-card selection states, since ordering / trigger-order / deck
        prompts often change ONLY those between steps."""
        if not state:
            return ""
        prompt_parts: list[tuple] = []
        for seat in ("player1", "player2"):
            prompt = (state.get("prompts") or {}).get(seat) or {}
            buttons = tuple(sorted(
                (str(b.get("text")), str(b.get("arg")), bool(b.get("disabled")))
                for b in (prompt.get("buttons") or [])
            ))
            displays = tuple(sorted(
                (str(d.get("cardUuid") or d.get("uuid")), str(d.get("selectionState", "")))
                for d in (prompt.get("displayCards") or []) if isinstance(d, dict)
            ))
            per_card = tuple(sorted(
                (str(b.get("text")), str(b.get("arg")))
                for b in (prompt.get("perCardButtons") or [])
            ))
            prompt_parts.append((
                seat,
                str(prompt.get("menuTitle", "")),
                tuple(sorted(str(u) for u in (prompt.get("selectableCards") or []))),
                tuple(sorted(str(u) for u in (prompt.get("selectedCards") or []))),
                buttons,
                displays,
                per_card,
            ))
        board = self._player_state_sig(state, str(state.get("player1Id") or ""))
        return f"{prompt_parts} || {board}"

    def _track_loop_repeat(self, action: dict[str, Any]) -> None:
        """Detect zero-progress action cycles and block the repeating action.

        A saturated policy can lock into e.g. click card → Cancel payment →
        click card → Cancel ... forever: every step costs -0.01, the state
        never changes, and the policy gradient dies (entropy 0, policy loss 0).
        If the same action is retried from an identical (prompt, board) state
        within the recent history, it is masked until the prompt moves on.
        """
        if not isinstance(action, dict) or not self.current_state:
            return
        state = self.current_state
        p_id = str(action.get("playerId") or "")
        if str(state.get("player1Id")) == p_id:
            p_key = "player1"
        elif str(state.get("player2Id")) == p_id:
            p_key = "player2"
        else:
            return
        prompt = (state.get("prompts") or {}).get(p_key) or {}
        menu_title = str(prompt.get("menuTitle", ""))
        if not menu_title or "waiting for opponent" in menu_title.lower():
            return
        # Include the prompt uuid so repeated prompts of the same title — e.g.
        # several queued optional triggers each asking "... or pass" — are NOT
        # mistaken for a no-progress loop. Same-instance retries still block.
        prompt_uuid = str(prompt.get("promptUuid", ""))
        selected = tuple(sorted(str(u) for u in prompt.get("selectedCards", [])))
        prompt_sig = (p_key, menu_title, selected, prompt_uuid)
        player_sig = self._player_state_sig(state, p_id)
        triple = (prompt_sig, player_sig, self._action_key(action))

        if triple in self._step_history:
            key = triple[2]
            if self._loop_blocks.get(key) != prompt_sig:
                self._loop_blocks[key] = prompt_sig
                if self._loop_block_print_budget > 0:
                    self._loop_block_print_budget -= 1
                    _sys.stderr.write(
                        f"[ENV] loop-block: masking {key!r} under {menu_title!r} "
                        f"(same no-progress step seen twice)\n"
                    )
                    _sys.stderr.flush()
            return
        self._step_history.append(triple)

    def _get_obs(self):
        """
        Encode the current game state as the structured SWU State Tensor.

        Layout (offsets defined by the module-level block constants):
          Block 1 [0..14)     Global, Force & Economic
            0  active player is the controlled player (1/0)
            1  controlled player holds initiative (1/0; this API serializes the
               initiative holder as `activePlayer`)
            2  phase hash (sum of char codes mod 100)
            3  friendly has Force token (1/0)
            4  enemy has Force token (1/0)
            5  friendly credits | 6 enemy credits
            7  friendly ready resources | 8 friendly exhausted resources
            9  enemy ready resources | 10 enemy exhausted resources
            11 friendly hand count | 12 enemy hand count | 13 reserved
          Block 2 [14..144)   Bases & Leaders
            friendly base  [maxHp, currentHp, epicActionAvailable] + 59 BASE_TRAITS
            enemy base     [maxHp, currentHp, epicActionAvailable] + 59 BASE_TRAITS
            friendly leader [isDeployed, epicAction/DeployUsed, isExhausted]
            enemy leader    [isDeployed, epicAction/DeployUsed, isExhausted]
          Block 3 [144..264) Friendly Hand — 10 slots × 12:
            [cost, printedPower, maxHp, type(1=unit/2=event/3=upgrade),
             playable, isUnique] + 6 ASPECTS multi-hot
          Block 4 [264..274) Opponent info-set densities (10 categories):
            (deck definition - discard pile) / remaining deck size
          Block 5 [274..2386) Dual arenas: ground-me, ground-opp, space-me,
            space-opp; 6 slots each; 88 floats per slot:
              0 occupied | 1 isLeaderUnit | 2 isTokenUnit | 3 isUnique
              4 printedCost | 5 currentPower | 6 maxHp | 7 currentHp
              8 isExhausted | 9 canAttackNow | 10 isLegalTarget
              11 shieldCount | 12 upgradeCount | 13 upgradePowerBonus
              14 upgradeHpBonus | 15 sentinel | 16 saboteur | 17 grit
              18 overwhelm | 19 ambush | 20 hidden/stealth | 21 raid
              22 restore | 23-28 ASPECTS | 29-87 UNIT_TRAITS
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        state = self.current_state
        if not state or "state" not in state:
            return obs

        my_key, opp_key = self._my_opp_keys(state)
        state_section = state.get("state") or {}
        my_state = state_section.get(my_key) or {}
        opp_state = state_section.get(opp_key) or {}

        # ── Block 1: global / force / economy ────────────────────────────────
        active_id = str(state.get("activePlayer") or "")
        obs[_OBS_B1 + 0] = 1.0 if active_id == str(self.player_id) else 0.0
        obs[_OBS_B1 + 1] = obs[_OBS_B1 + 0]
        phase = str(state.get("phase") or "")
        obs[_OBS_B1 + 2] = float(sum(ord(ch) for ch in phase) % 100)
        obs[_OBS_B1 + 3] = 1.0 if my_state.get("hasForceToken") else 0.0
        obs[_OBS_B1 + 4] = 1.0 if opp_state.get("hasForceToken") else 0.0
        obs[_OBS_B1 + 5] = self._num(my_state, "credits")
        obs[_OBS_B1 + 6] = self._num(opp_state, "credits")
        obs[_OBS_B1 + 7] = self._num(my_state, "readyResourceCount")
        obs[_OBS_B1 + 8] = self._num(my_state, "exhaustedResourceCount")
        obs[_OBS_B1 + 9] = self._num(opp_state, "readyResourceCount")
        obs[_OBS_B1 + 10] = self._num(opp_state, "exhaustedResourceCount")
        obs[_OBS_B1 + 11] = float(len(my_state.get("hand") or []))
        obs[_OBS_B1 + 12] = float(len(opp_state.get("hand") or []))
        obs[_OBS_B1 + 13] = 0.0  # reserved

        # ── Block 2: bases & leaders ──────────────────────────────────────────
        def _write_base(offset: int, base_state: dict[str, Any]) -> None:
            db = self._card_data(base_state.get("internalName"))
            obs[offset] = self._max_hp(base_state, db)
            obs[offset + 1] = self._current_hp(base_state, db)
            obs[offset + 2] = 1.0 if db.get("epicAction") else 0.0
            obs[offset + 3:offset + 3 + len(BASE_TRAITS)] = self._trait_hot(
                db.get("traits"), _BASE_TRAIT_INDEX, len(BASE_TRAITS)
            )

        offset = _OBS_B2
        _write_base(offset, my_state.get("base") or {})
        offset += 3 + len(BASE_TRAITS)
        _write_base(offset, opp_state.get("base") or {})
        offset += 3 + len(BASE_TRAITS)

        def _write_leader(offset: int, leader_state: dict[str, Any]) -> None:
            db = self._card_data(leader_state.get("internalName"))
            zone = self._zone_name(leader_state)
            deployed = zone in ("ground", "space")
            obs[offset] = 1.0 if deployed else 0.0
            # Leaders deploy by flipping; if deployed and the card has an epic
            # action, the deploy/epic-action side is treated as used.
            obs[offset + 1] = 1.0 if (deployed and bool(db.get("epicAction"))) else 0.0
            obs[offset + 2] = 1.0 if self._is_exhausted(leader_state) else 0.0

        _write_leader(offset, my_state.get("leader") or {})
        offset += 3
        _write_leader(offset, opp_state.get("leader") or {})
        offset += 3

        # ── Block 3: friendly hand ────────────────────────────────────────────
        playable_map = self._debug_playable_map(state)

        def _hand_cost(card: dict[str, Any]) -> float:
            cost = self._card_data(card.get("internalName")).get("cost")
            return float(cost) if cost is not None else 999.0

        my_hand = sorted(my_state.get("hand") or [], key=_hand_cost, reverse=True)
        ready_resources = self._num(my_state, "readyResourceCount")
        for slot in range(HAND_SLOTS):
            base = _OBS_B3 + slot * HAND_FEATURES
            if slot >= len(my_hand):
                continue
            card = my_hand[slot]
            db = self._card_data(card.get("internalName"))
            cost = db.get("cost")
            printed_power = db.get("power")
            if printed_power is None:
                printed_power = self._num(card, "power", "printedPower")
            obs[base + 0] = float(cost) if cost is not None else 0.0
            obs[base + 1] = float(printed_power or 0.0)
            obs[base + 2] = self._max_hp(card, db)
            obs[base + 3] = self._card_type_scalar(db)
            can_play = playable_map.get(str(card.get("internalName") or "").lower())
            if can_play is None:
                can_play = cost is not None and float(cost) <= ready_resources
            obs[base + 4] = 1.0 if can_play else 0.0
            obs[base + 5] = 1.0 if db.get("unique") else 0.0
            obs[base + 6:base + 12] = self._aspect_hot(db.get("aspects"))

        # ── Block 4: opponent info-set densities ─────────────────────────────
        obs[_OBS_B4:_OBS_B4 + BLOCK4_SIZE] = self._opponent_deck_densities(opp_key, opp_state)

        # ── Block 5: dual arena unit matrices ─────────────────────────────────
        legal_target_uuids: set[str] = set()
        for seat in ("player1", "player2"):
            prompt = (state.get("prompts") or {}).get(seat) or {}
            for uuid in prompt.get("selectableCards") or []:
                legal_target_uuids.add(str(uuid))
            for display_card in prompt.get("displayCards") or []:
                if isinstance(display_card, dict):
                    uuid = display_card.get("cardUuid") or display_card.get("uuid")
                    if uuid:
                        legal_target_uuids.add(str(uuid))

        def _arena_units(player_state: dict[str, Any], arena_key: str) -> list[dict[str, Any]]:
            units = list(player_state.get(arena_key) or [])
            leader = player_state.get("leader")
            if isinstance(leader, dict):
                want = "ground" if arena_key == "groundArena" else "space"
                if self._zone_name(leader) == want and all(
                    str(unit.get("uuid")) != str(leader.get("uuid")) for unit in units
                ):
                    units = units + [leader]
            return units

        def _encode_slot(base: int, unit: dict[str, Any] | None) -> None:
            if not isinstance(unit, dict) or not unit.get("uuid"):
                return
            obs[base + 0] = 1.0
            db = self._card_data(unit.get("internalName"))
            types = {str(t).lower() for t in (db.get("types") or [])}
            keywords = {str(k).lower() for k in (db.get("keywords") or [])}
            obs[base + 1] = 1.0 if "leader" in types else 0.0
            obs[base + 2] = 1.0 if "token" in types else 0.0
            obs[base + 3] = 1.0 if db.get("unique") else 0.0
            obs[base + 4] = float(db.get("cost") or 0)
            obs[base + 5] = self._num(unit, "power", "printedPower")
            obs[base + 6] = self._max_hp(unit, db)
            obs[base + 7] = self._current_hp(unit, db)
            exhausted = self._is_exhausted(unit)
            zone = self._zone_name(unit)
            ready = (not exhausted) and zone in ("ground", "space")
            obs[base + 8] = 1.0 if exhausted else 0.0
            obs[base + 9] = 1.0 if (ready and obs[base + 5] > 0.0) else 0.0
            obs[base + 10] = 1.0 if str(unit.get("uuid")) in legal_target_uuids else 0.0
            # The serialized state does not expose live shield tokens; "Shielded"
            # grants one shield on entry, so encode the floor we can observe.
            obs[base + 11] = 1.0 if "shielded" in keywords else 0.0
            upgrades = unit.get("upgrades") or []
            obs[base + 12] = float(len(upgrades))
            upgrade_power = 0.0
            upgrade_hp = 0.0
            for upgrade in upgrades:
                upgrade_db = self._card_data(upgrade.get("internalName"))
                upgrade_power += float(upgrade_db.get("upgradePower") or 0)
                upgrade_hp += float(upgrade_db.get("upgradeHp") or 0)
            obs[base + 13] = upgrade_power
            obs[base + 14] = upgrade_hp
            flags = self._keyword_flags(db)
            obs[base + 15] = 1.0 if flags.get("sentinel") else 0.0
            obs[base + 16] = 1.0 if flags.get("saboteur") else 0.0
            obs[base + 17] = 1.0 if flags.get("grit") else 0.0
            obs[base + 18] = 1.0 if flags.get("overwhelm") else 0.0
            obs[base + 19] = 1.0 if flags.get("ambush") else 0.0
            obs[base + 20] = 1.0 if flags.get("hidden") else 0.0
            obs[base + 21] = float(flags.get("raid") or 0.0)
            obs[base + 22] = float(flags.get("restore") or 0.0)
            obs[base + 23:base + 23 + len(ASPECTS)] = self._aspect_hot(db.get("aspects"))
            obs[base + 29:base + 29 + len(UNIT_TRAITS)] = self._trait_hot(
                db.get("traits"), _UNIT_TRAIT_INDEX, len(UNIT_TRAITS)
            )

        group_specs = (
            (my_state, "groundArena"),
            (opp_state, "groundArena"),
            (my_state, "spaceArena"),
            (opp_state, "spaceArena"),
        )
        for group, (player_state, arena_key) in enumerate(group_specs):
            units = _arena_units(player_state, arena_key)
            for slot in range(ARENA_SLOTS_PER_SIDE):
                unit = units[slot] if slot < len(units) else None
                _encode_slot(
                    _OBS_B5
                    + group * ARENA_SLOTS_PER_SIDE * UNIT_SLOT_FEATURES
                    + slot * UNIT_SLOT_FEATURES,
                    unit,
                )

        return obs

    def print_agent_knowledge(self) -> None:
        """Human-readable dump of everything the agent currently observes / may act on."""
        state = self.current_state
        if not state or "state" not in state:
            print("<no state — call reset()/refresh() first>")
            return

        my_key, opp_key = self._my_opp_keys(state)
        state_section = state.get("state") or {}
        my_state = state_section.get(my_key) or {}
        opp_state = state_section.get(opp_key) or {}

        print("=" * 78)
        print(f"[Agent Knowledge] player={self.player_id} ({my_key})  phase={state.get('phase')}")
        active = str(state.get("activePlayer") or "")
        print(f"  active/initiative : {active or '?'} {'(ME)' if active == str(self.player_id) else '(opponent)'}")
        print(f"  force token       : me={1 if my_state.get('hasForceToken') else 0}  "
              f"opp={1 if opp_state.get('hasForceToken') else 0}")
        print(f"  resources         : me {my_state.get('readyResourceCount', 0)} ready / "
              f"{my_state.get('exhaustedResourceCount', 0)} exhausted | "
              f"opp {opp_state.get('readyResourceCount', 0)} ready / "
              f"{opp_state.get('exhaustedResourceCount', 0)} exhausted")
        print(f"  credits           : me={my_state.get('credits', 0)}  opp={opp_state.get('credits', 0)}")
        print(f"  hand / deck       : me {len(my_state.get('hand') or [])} hand / "
              f"{len(my_state.get('deck') or [])} deck | "
              f"opp {len(opp_state.get('hand') or [])} hand / {len(opp_state.get('deck') or [])} deck")

        for label, player_state in (("ME ", my_state), ("OPP", opp_state)):
            base_state = player_state.get("base") or {}
            base_db = self._card_data(base_state.get("internalName"))
            base_traits = ", ".join(
                trait for trait in (base_db.get("traits") or [])
                if str(trait).lower() in _BASE_TRAIT_INDEX
            ) or "—"
            print(f"  {label} BASE   {base_state.get('internalName', '?')} "
                  f"HP={self._current_hp(base_state, base_db):.0f}/{self._max_hp(base_state, base_db):.0f} "
                  f"epicAction={'yes' if base_db.get('epicAction') else 'no'} traits=[{base_traits}]")
            leader_state = player_state.get("leader") or {}
            leader_db = self._card_data(leader_state.get("internalName"))
            leader_zone = self._zone_name(leader_state)
            print(f"  {label} LEADER {leader_state.get('internalName', '?')} "
                  f"deployed={'yes' if leader_zone in ('ground', 'space') else 'no'} "
                  f"exhausted={'yes' if self._is_exhausted(leader_state) else 'no'} "
                  f"HP={self._current_hp(leader_state, leader_db):.0f}/{self._max_hp(leader_state, leader_db):.0f}")

        playable_map = self._debug_playable_map(state)

        def _hand_cost(card: dict[str, Any]) -> float:
            cost = self._card_data(card.get("internalName")).get("cost")
            return float(cost) if cost is not None else 999.0

        print("  HAND:")
        for card in sorted(my_state.get("hand") or [], key=_hand_cost, reverse=True):
            db = self._card_data(card.get("internalName"))
            can_play = playable_map.get(str(card.get("internalName") or "").lower())
            status = "yes" if can_play else ("no" if can_play is not None else "?")
            print(f"    - {card.get('internalName', '?')} cost={db.get('cost')} "
                  f"type={'/'.join(db.get('types') or [])} playable={status}")

        densities = self._opponent_deck_densities(opp_key, opp_state)
        print("  OPPONENT UNSEEN-THREAT DENSITIES ((deck def - discard) / remaining):")
        for category, label in enumerate(DECK_CATEGORY_LABELS):
            print(f"    {label:<18} {densities[category]:.3f}")

        for arena_label, arena_key in (("GROUND ARENA", "groundArena"), ("SPACE ARENA", "spaceArena")):
            print(f"  {arena_label}:")
            for owner_label, player_state in (("me ", my_state), ("opp", opp_state)):
                units = list(player_state.get(arena_key) or [])
                leader_state = player_state.get("leader")
                want = "ground" if arena_key == "groundArena" else "space"
                if (isinstance(leader_state, dict) and self._zone_name(leader_state) == want
                        and all(str(unit.get("uuid")) != str(leader_state.get("uuid")) for unit in units)):
                    units = units + [leader_state]
                if not units:
                    print(f"    [{owner_label}] (empty)")
                    continue
                for unit in units:
                    db = self._card_data(unit.get("internalName"))
                    flags = self._keyword_flags(db)
                    active_keywords = [
                        keyword for keyword in ("sentinel", "saboteur", "grit", "overwhelm", "ambush", "hidden")
                        if flags.get(keyword)
                    ]
                    if flags.get("raid"):
                        active_keywords.append(f"raid {flags['raid']:.0f}")
                    if flags.get("restore"):
                        active_keywords.append(f"restore {flags['restore']:.0f}")
                    print(f"    [{owner_label}] {unit.get('internalName', '?')} "
                          f"P={self._num(unit, 'power', 'printedPower'):.0f} "
                          f"HP={self._current_hp(unit, db):.0f}/{self._max_hp(unit, db):.0f} "
                          f"exhausted={'yes' if self._is_exhausted(unit) else 'no'} "
                          f"kw=[{','.join(active_keywords) or '—'}]")

        mask = self.legal_action_mask
        total = len(self.available_actions)
        legal = int(mask.sum()) if mask is not None else total
        print(f"  ACTIONS: {total} available, {legal} mask-enabled")
        for index, action in enumerate(self.available_actions):
            allowed = "OK  " if (mask is not None and bool(mask[index])) else "MASK"
            label = action.get("internalName") or action.get("promptText") or action.get("arg") or "?"
            print(f"    [{index}] {allowed} {action.get('actionType', '?')} {label}")
        print("=" * 78)

    def _get_info(self):
        active_prompts = []
        if self.current_state and "prompts" in self.current_state:
            for p, pdata in self.current_state["prompts"].items():
                if pdata and "Waiting" not in pdata.get("menuTitle", ""):
                    active_prompts.append(f"{p}: '{pdata.get('menuTitle')}'")
        
        return {
            "phase": self.current_state.get("phase") if self.current_state else None,
            "activePlayer": self.active_player,
            "no_progress": bool(getattr(self, "_last_step_no_progress", False)),
            "activePlayers": self.active_players,
            "activePrompts": active_prompts,
            "num_valid_actions": len(self.available_actions),
            "num_legal_actions": int(self.legal_action_mask.sum()) if self.legal_action_mask is not None else 0,
            "state_dict": self.current_state
        }


    def sync_state(self, state):
        """Replace the cached server state and rebuild the available action list."""
        self.current_state = state
        self._update_available_actions()

    def _build_distribution_results_for_prompt(self, prompt_key: str) -> dict[str, Any] | None:
        if not self.current_state or "prompts" not in self.current_state:
            return None

        prompt_state = self.current_state["prompts"].get(prompt_key) or {}
        distribute_prompt = prompt_state.get("distributeAmongTargets") or {}
        amount = distribute_prompt.get("amount", 0)
        distribution_type = distribute_prompt.get("type")

        if amount <= 0 or not distribution_type:
            return None

        state_section = self.current_state.get("state") or {}
        player_state = state_section.get(prompt_key) or {}

        def _remaining_hp(card: dict[str, Any]) -> float:
            if not isinstance(card, dict):
                return 0.0

            if card.get("remainingHp") is not None:
                try:
                    return max(0.0, float(card.get("remainingHp")))
                except Exception:
                    return 0.0

            if card.get("currentHp") is not None:
                try:
                    return max(0.0, float(card.get("currentHp")))
                except Exception:
                    return 0.0

            try:
                hp_value = float(card.get("hp") or 0.0)
                damage_value = float(card.get("damage") or 0.0)
                if hp_value > 0:
                    return max(0.0, hp_value - damage_value)
            except Exception:
                pass

            return 0.0

        def _is_unit(card: dict[str, Any]) -> bool:
            try:
                return bool(card.get("isUnit")) or card.get("power") is not None or card.get("printedPower") is not None
            except Exception:
                return False

        def _collect_cards_by_uuid() -> dict[str, dict[str, Any]]:
            cards: dict[str, dict[str, Any]] = {}

            def _add_card(card: dict[str, Any] | None) -> None:
                if not isinstance(card, dict):
                    return
                uuid = card.get("uuid")
                if uuid:
                    cards[str(uuid)] = card

            for zone in ("hand", "spaceArena", "groundArena"):
                for card in player_state.get(zone, []):
                    _add_card(card)
                    for upgrade in card.get("upgrades", []):
                        _add_card(upgrade)
            for key in ("leader", "base"):
                _add_card(player_state.get(key))
                for upgrade in (player_state.get(key) or {}).get("upgrades", []):
                    _add_card(upgrade)

            opp_key = "player2" if prompt_key == "player1" else "player1"
            opp_state = state_section.get(opp_key) or {}
            for zone in ("spaceArena", "groundArena"):
                for card in opp_state.get(zone, []):
                    _add_card(card)
                    for upgrade in card.get("upgrades", []):
                        _add_card(upgrade)
            for key in ("leader", "base"):
                _add_card(opp_state.get(key))
                for upgrade in (opp_state.get(key) or {}).get("upgrades", []):
                    _add_card(upgrade)

            return cards

        cards_by_uuid = _collect_cards_by_uuid()

        def _resolve_prompt_card(card_ref: dict[str, Any]) -> dict[str, Any] | None:
            uuid = card_ref.get("uuid") or card_ref.get("cardUuid")
            if uuid and str(uuid) in cards_by_uuid:
                return cards_by_uuid[str(uuid)]
            if uuid and isinstance(card_ref, dict):
                return card_ref
            return None

        candidate_cards = []
        display_cards = prompt_state.get("displayCards") or []
        selectable_uuids = {str(uuid) for uuid in (prompt_state.get("selectableCards") or []) if uuid}

        if display_cards:
            for card_ref in display_cards:
                if not isinstance(card_ref, dict) or card_ref.get("selectionState") == "invalid":
                    continue
                ref_uuid = card_ref.get("uuid") or card_ref.get("cardUuid")
                if selectable_uuids and str(ref_uuid) not in selectable_uuids:
                    continue
                resolved = _resolve_prompt_card(card_ref)
                if resolved:
                    candidate_cards.append(resolved)
        elif selectable_uuids:
            for uuid in selectable_uuids:
                card = cards_by_uuid.get(uuid)
                if card:
                    candidate_cards.append(card)
        else:
            for card in cards_by_uuid.values():
                candidate_cards.append(card)

        if not candidate_cards:
            if distribute_prompt.get("canChooseNoTargets"):
                return {"type": distribution_type, "valueDistribution": []}
            return None

        value_distribution = []

        remaining = int(amount)
        max_targets = distribute_prompt.get("maxTargets")
        # Absorb the damage on the highest-remaining-HP targets first (typically
        # a base at 25-30 HP): concentrating damage minimizes unit losses.
        candidate_cards.sort(key=lambda card: _remaining_hp(card), reverse=True)

        is_indirect_damage = distribution_type == "distributeIndirectDamage"

        for card in candidate_cards:
            if remaining <= 0:
                break
            if max_targets is not None and len(value_distribution) >= int(max_targets):
                break

            cap = remaining
            if is_indirect_damage and _is_unit(card):
                cap = min(cap, int(_remaining_hp(card)))

            if cap <= 0:
                continue

            value_distribution.append({"uuid": card.get("uuid") or card.get("cardUuid"), "amount": cap})
            remaining -= cap

        if remaining > 0 and not distribute_prompt.get("canDistributeLess"):
            # We could not find a full legal allocation under the target cap constraints.
            # Leave the results empty so the caller can retry with a different prompt state.
            return None

        if not value_distribution and not distribute_prompt.get("canChooseNoTargets"):
            return None

        if is_indirect_damage:
            for entry in value_distribution:
                card = cards_by_uuid.get(str(entry.get("uuid")))
                if card and _is_unit(card) and int(entry.get("amount") or 0) > int(_remaining_hp(card)):
                    return None

        return {"type": distribution_type, "valueDistribution": value_distribution}

    def _state_player_key(self, state: dict[str, Any] | None) -> str | None:
        if not state:
            return None
        if str(state.get("player1Id")) == str(self.player_id):
            return "player1"
        if str(state.get("player2Id")) == str(self.player_id):
            return "player2"
        return None

    def _safe_state_player(self, state: dict[str, Any] | None, key: str | None) -> dict[str, Any]:
        if not state or not key:
            return {}
        return (state.get("state") or {}).get(key, {})

    def _sum_board_power(self, player_state: dict[str, Any]) -> float:
        total = 0.0
        for zone in ("spaceArena", "groundArena"):
            for card in player_state.get(zone, []):
                total += float(card.get("power") or card.get("printedPower") or 0.0)
        return total

    def _sum_board_hp(self, player_state: dict[str, Any]) -> float:
        total = 0.0
        for zone in ("spaceArena", "groundArena"):
            for card in player_state.get(zone, []):
                total += float(card.get("hp") or card.get("remainingHp") or card.get("currentHp") or 0.0)
        return total

    def _get_base_hp(self, player_state: dict[str, Any]) -> float:
        base = player_state.get("base") or {}
        return float(base.get("hp") or base.get("remainingHp") or base.get("currentHp") or base.get("maxHp") or 0.0)

    def _get_leader_hp(self, player_state: dict[str, Any]) -> float:
        leader = player_state.get("leader") or {}
        return float(leader.get("hp") or leader.get("remainingHp") or leader.get("currentHp") or leader.get("maxHp") or 0.0)

    def _count_units(self, player_state: dict[str, Any]) -> float:
        total = 0.0
        for zone in ("spaceArena", "groundArena"):
            total += float(len(player_state.get(zone, [])))
        return total

    def _reward_pass_and_resource_usage(self, prev_state: dict[str, Any] | None, current_state: dict[str, Any] | None, action_dict: dict[str, Any]) -> float:
        if not prev_state or not current_state:
            return 0.0

        prev_key = self._state_player_key(prev_state)
        curr_key = self._state_player_key(current_state)
        if not prev_key or not curr_key:
            return 0.0

        prev_player = self._safe_state_player(prev_state, prev_key)
        curr_player = self._safe_state_player(current_state, curr_key)

        prev_ready = float(prev_player.get("readyResourceCount") or 0.0)
        curr_ready = float(curr_player.get("readyResourceCount") or 0.0)
        prev_credits = float(prev_player.get("credits") or 0.0)
        curr_credits = float(curr_player.get("credits") or 0.0)

        action_text = str(action_dict.get("promptText", "")).strip().lower()
        action_arg = str(action_dict.get("arg", "")).strip().lower()
        full_text = f"{action_text} {action_arg}"

        reward = 0.0

        # Encourage actually spending available resources.
        resource_delta = (prev_ready - curr_ready) + 0.5 * (prev_credits - curr_credits)
        reward += 0.04 * resource_delta

        # Small reward for playing a card (any clickCard action) — encourages
        # the agent to use its hand rather than hoarding cards.
        # if str(action_dict.get("actionType", "")).lower() == "clickcard":
        #     reward += 0.05

        # Penalty for the agent passing with leftover resources OR unexhausted units.
        # Only applies when the agent passes (not when claiming initiative).
        action_player_id = str(action_dict.get("playerId", ""))
        is_agent_action = action_player_id == str(self.player_id)
        is_claim = "claim" in full_text
        is_pass = (not is_claim) and ("pass" in full_text)

        if is_agent_action and is_pass:
            # Relative waste: penalty scales with fraction of total capacity left unspent.
            prev_exhausted = float(prev_player.get("exhaustedResourceCount") or 0.0)
            total_capacity = prev_ready + prev_exhausted + 0.5 * prev_credits
            wasted = prev_ready + 0.5 * prev_credits
            waste_frac = wasted / max(1.0, total_capacity) if total_capacity > 0 else 0.0
            if waste_frac > 0.1:
                reward -= min(1.0, waste_frac * 1.2)
            # Count unexhausted (ready) units on the agent's board.
            curr_player_state = self._safe_state_player(current_state, curr_key)
            unexhausted = 0.0
            for zone in ("spaceArena", "groundArena"):
                for card in curr_player_state.get(zone, []):
                    if not (card.get("exhausted") or card.get("isExhausted") or card.get("is_exhausted")):
                        unexhausted += 1.0
            if unexhausted > 0:
                reward -= min(1.2, 0.3 * unexhausted)
            # Extra penalty for plain "Pass" vs "Claim initiative" — passing gives
            # the opponent a free turn without gaining initiative. Passing is
            # almost always the worst action, so make it clearly expensive.
            is_claim = "claim" in full_text
            if not is_claim:
                reward -= 0.5

        return reward

    def _shape_reward(self, prev_state: dict[str, Any] | None, current_state: dict[str, Any] | None, action_dict: dict[str, Any]) -> float:
        if not current_state:
            return 0.0

        reward = self._reward_pass_and_resource_usage(prev_state, current_state, action_dict)

        # Per-step penalty: every step costs, pushing the agent to close games efficiently.
        # At 0.02/step, 200 steps = -4.0, 500 steps = -10.0 — makes prolonging costly.
        step_penalty = -0.01
        reward += step_penalty

        opp_base_r = 0.0
        opp_leader_r = 0.0
        my_base_r = 0.0
        my_leader_r = 0.0
        board_power_r = 0.0
        board_hp_r = 0.0
        kill_r = 0.0
        loss_r = 0.0
        win_r = 0.0

        prev_key = self._state_player_key(prev_state)
        curr_key = self._state_player_key(current_state)
        if prev_key and curr_key:
            prev_player = self._safe_state_player(prev_state, prev_key)
            curr_player = self._safe_state_player(current_state, curr_key)
            opp_key = "player2" if curr_key == "player1" else "player1"
            prev_opp = self._safe_state_player(prev_state, opp_key)
            curr_opp = self._safe_state_player(current_state, opp_key)

            prev_opp_base = self._get_base_hp(prev_opp)
            curr_opp_base = self._get_base_hp(curr_opp)
            prev_opp_leader = self._get_leader_hp(prev_opp)
            curr_opp_leader = self._get_leader_hp(curr_opp)
            prev_my_base = self._get_base_hp(prev_player)
            curr_my_base = self._get_base_hp(curr_player)
            prev_my_leader = self._get_leader_hp(prev_player)
            curr_my_leader = self._get_leader_hp(curr_player)

            prev_my_power = self._sum_board_power(prev_player)
            curr_my_power = self._sum_board_power(curr_player)
            prev_opp_power = self._sum_board_power(prev_opp)
            curr_opp_power = self._sum_board_power(curr_opp)

            prev_my_hp = self._sum_board_hp(prev_player)
            curr_my_hp = self._sum_board_hp(curr_player)
            prev_opp_hp = self._sum_board_hp(prev_opp)
            curr_opp_hp = self._sum_board_hp(curr_opp)

            prev_my_units = self._count_units(prev_player)
            curr_my_units = self._count_units(curr_player)
            prev_opp_units = self._count_units(prev_opp)
            curr_opp_units = self._count_units(curr_opp)

            # Opponent base/leader damage — the main path to winning.
            opp_base_r = 0.30 * max(0.0, prev_opp_base - curr_opp_base)
            reward += opp_base_r

            opp_leader_r = 0.20 * max(0.0, prev_opp_leader - curr_opp_leader)
            reward += opp_leader_r

            # Losing our own base/leader is very bad.
            my_base_r = -0.30 * max(0.0, prev_my_base - curr_my_base)
            reward += my_base_r

            my_leader_r = -0.20 * max(0.0, prev_my_leader - curr_my_leader)
            reward += my_leader_r

            # Board advantage.
            board_power_delta = (curr_my_power - prev_my_power) - (curr_opp_power - prev_opp_power)
            board_power_r = 0.03 * board_power_delta
            reward += board_power_r

            board_hp_delta = (curr_my_hp - prev_my_hp) - (curr_opp_hp - prev_opp_hp)
            board_hp_r = 0.015 * board_hp_delta
            reward += board_hp_r

            # Card kills / losses — make unit trading worthwhile.
            # A typical unit has 3-5 HP; killing it should be comparable to
            # dealing that much base damage (0.30/HP × 4 = 1.2).
            kill_r = 1.0 * max(0.0, prev_opp_units - curr_opp_units)
            reward += kill_r
            loss_r = -0.5 * max(0.0, prev_my_units - curr_my_units)
            reward += loss_r

        winners = current_state.get("winners", []) if current_state else []
        phase = current_state.get("phase") if current_state else None

        # Terminal reward: use base HP to determine winner (winners list is unreliable).
        # Recompute current player keys from current_state to avoid scoping issues.
        term_key = self._state_player_key(current_state)
        if term_key:
            term_player = self._safe_state_player(current_state, term_key)
            term_opp_key = "player2" if term_key == "player1" else "player1"
            term_opp = self._safe_state_player(current_state, term_opp_key)
            my_base = self._get_base_hp(term_player)
            opp_base = self._get_base_hp(term_opp)
            if opp_base <= 0 and my_base > 0:
                win_r = 10.0   # agent won
                reward += win_r
            elif my_base <= 0 and opp_base > 0:
                win_r = -10.0  # agent lost
                reward += win_r
            elif my_base <= 0 and opp_base <= 0:
                win_r = 0.0    # simultaneous destruction — draw
        elif len(winners) > 0 or phase == "game_end":
            # Fallback if player-key lookup fails but winners list populated
            win_r = 5.0 if str(self.player_id) in {str(w) for w in winners} else -5.0
            reward += win_r

        # Debug: print reward breakdown every 50th call
        # if not hasattr(self, "_reward_debug_count"):
        #     self._reward_debug_count = 0
        # self._reward_debug_count += 1
        # if self._reward_debug_count % 50 == 0:
        #     print(
        #         f"[RWD dbg] step={step_penalty:+.3f} "
        #         f"oppBase={opp_base_r:+.3f} oppLead={opp_leader_r:+.3f} "
        #         f"myBase={my_base_r:+.3f} myLead={my_leader_r:+.3f} "
        #         f"pow={board_power_r:+.3f} hp={board_hp_r:+.3f} "
        #         f"kill={kill_r:+.3f} loss={loss_r:+.3f} "
        #         f"win={win_r:+.3f} | total={reward:+.3f}"
        #     )

        return float(reward)
