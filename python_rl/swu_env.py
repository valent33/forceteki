import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import warnings
import copy
from typing import Any

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

        # Describe the observation space: 
        # For simplicity, we flatten the game state into a massive feature Box.
        # In a deep architecture, this is better represented as Dict of sequences (transformers).
        self.observation_space = spaces.Box(low=-10.0, high=100.0, shape=(64,), dtype=np.float32)

        self.current_state = None
        self.available_actions = []
        self.active_players = []
        # Track card UUIDs already clicked in multi-select prompts so the agent
        # can't keep picking the same cards — exhausts all options, then must click "Done".
        self._consumed_card_uuids: set[str] = set()
        self._last_prompt_key: str | None = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # POST /reset connects to Node.js backend
        payload = options if options else {
            "options": {
                "phase": "action",
                "player1": {"hasInitiative": True}
            }
        }
        
        resp = requests.post(f"{self.server_url}/reset", json=payload)
        resp.raise_for_status()

        self.current_state = resp.json()
        if "error" in self.current_state:
            raise RuntimeError(f"Error from server on reset: {self.current_state['error']}")

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

    def step(self, action_index):
        # Prevent invalid out-of-bounds actions
        if action_index >= len(self.available_actions):
            warnings.warn(f"Agent chose invalid action index {action_index}. Max is {len(self.available_actions)-1}.")
            # Penalize agent heavily for an illegal action and optionally terminate
            return self._get_obs(), -1.0, False, False, self._get_info()

        prev_state = copy.deepcopy(self.current_state)
        action_dict = self.available_actions[action_index]
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
            resp = requests.post(f"{self.server_url}/step", json=payload)
            resp.raise_for_status()
            self.current_state = resp.json()
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
                    # print(f"[DEBUG] Game Over! Winners: {winners}")
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
                            # print(f"[DEBUG] Game Over detected from prompt: {title}")
                            break

                self._update_available_actions()

            return self._get_obs(), reward, terminated, truncated, self._get_info()

        if action_dict["actionType"] == "macro_resource_cards":
            # 1. Click the cards
            for uid in action_dict.get("uuids", []):
                payload_card = {
                    "playerId": action_dict.get("playerId", self.player_id),
                    "action": "clickCard",
                    "arg": "any",
                    "uuid": uid,
                    "method": "undefined",
                    "promptText": ""
                }
                requests.post(f"{self.server_url}/step", json=payload_card).raise_for_status()

            # 2. Refresh and click the actual Done button metadata from current prompt.
            # Some prompts require specific arg/uuid/method values and can stall otherwise.
            state_resp = requests.get(f"{self.server_url}/state")
            state_resp.raise_for_status()
            state = state_resp.json()

            p_key = "player1" if action_dict.get("playerId") == "111" else "player2"
            prompt = (state.get("prompts") or {}).get(p_key) or {}
            done_btn = None
            for btn in prompt.get("buttons", []):
                text = str(btn.get("text", "")).strip().lower()
                arg = str(btn.get("arg", "")).strip().lower()
                if text == "done" or arg == "done":
                    done_btn = btn
                    break

            if done_btn is not None:
                payload_done = {
                    "playerId": action_dict.get("playerId", self.player_id),
                    "action": "clickPrompt",
                    "arg": done_btn.get("arg", "done"),
                    "uuid": done_btn.get("uuid", ""),
                    "method": done_btn.get("command", "menuButton"),
                    "promptText": done_btn.get("text", "Done")
                }
                resp = requests.post(f"{self.server_url}/step", json=payload_done)
                resp.raise_for_status()
                self.current_state = resp.json()
            else:
                # If there is no done button anymore, card clicks likely auto-submitted.
                self.current_state = state
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
            resp = requests.post(f"{self.server_url}/step", json=payload)
            resp.raise_for_status()
            self.current_state = resp.json()
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
            resp = requests.post(f"{self.server_url}/step", json=payload)
            resp.raise_for_status()
            self.current_state = resp.json()
        else:
            payload = {
                "playerId": action_dict.get("playerId", self.player_id),
                "action": action_dict["actionType"],
                "arg": action_dict.get("arg", "menuButton"),
                "uuid": action_dict.get("uuid"),
                "method": action_dict.get("method"),
                "promptText": action_dict.get("promptText")
            }

            # Issue the action
            resp = requests.post(f"{self.server_url}/step", json=payload)
            resp.raise_for_status()
            self.current_state = resp.json()

        # Track consumed card UUID — prevents re-selection in multi-select prompts.
        if action_dict.get("actionType") in {"clickCard", "displayCardClick"} and action_dict.get("uuid"):
            self._consumed_card_uuids.add(str(action_dict["uuid"]))
        elif action_dict.get("actionType") == "perCardMenuButton" and action_dict.get("cardUuid"):
            self._consumed_card_uuids.add(str(action_dict["cardUuid"]))

        reward = self._shape_reward(prev_state, self.current_state, action_dict)
        terminated = False
        truncated = False

        if "error" in self.current_state:
            # print(f"Error on step: {self.current_state['error']}")
            # Treat server crash/illegal logic failure as an episode end
            terminated = True
            reward = -10.0
        else:
            # check the winners field outputted by the server
            winners = self.current_state.get("winners", [])
            if len(winners) > 0:
                terminated = True
                # print(f"[DEBUG] Game Over! Winners: {winners}")
            elif self.current_state.get("phase") == "game_end": # fallback
                terminated = True
            else:
                # Some builds report the end-of-game via a prompt message instead of winners list.
                prompts = self.current_state.get("prompts") or {}
                for p in prompts.values():
                    if not p:
                        continue
                    title = str(p.get("menuTitle", "")).lower()
                    if "has won" in title or "has won the game" in title:
                        terminated = True
                        # print(f"[DEBUG] Game Over detected from prompt: {title}")
                        break

            self._update_available_actions()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _update_available_actions(self):
        """
        Translates the current UI prompt / clickable states on the Node.js side 
        into a flat list of valid actions.
        """
        self.available_actions = []
        self.active_player = None
        self.active_players = []
        
        if not self.current_state or "prompts" not in self.current_state:
            return

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
                    if selection_state not in {"viewonly", "invalid"}:
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
                if title and "waiting for opponent" not in title.lower() and (has_buttons or has_dropdowns or has_cards):
                    focus_key = candidate_key
            elif active_player_id == str(self.current_state.get("player2Id")):
                candidate_key = "player2"
                candidate_prompt = self.current_state["prompts"].get(candidate_key) or {}
                title = str(candidate_prompt.get("menuTitle", ""))
                has_buttons = len(candidate_prompt.get("buttons", [])) > 0
                has_dropdowns = len(candidate_prompt.get("dropdownListOptions", [])) > 0
                has_cards = _has_interactive_display_cards(candidate_prompt)
                if title and "waiting for opponent" not in title.lower() and (has_buttons or has_dropdowns or has_cards):
                    focus_key = candidate_key

            # If the active player's prompt is not actionable yet, fall back to whichever prompt is actually asking for a decision.
            for candidate_key in ("player1", "player2"):
                if focus_key is not None:
                    break
                candidate_prompt = self.current_state["prompts"].get(candidate_key) or {}
                title = str(candidate_prompt.get("menuTitle", ""))
                if title and "waiting for opponent" not in title.lower():
                    has_buttons = len(candidate_prompt.get("buttons", [])) > 0
                    has_dropdowns = len(candidate_prompt.get("dropdownListOptions", [])) > 0
                    has_cards = _has_interactive_display_cards(candidate_prompt)
                    if has_buttons or has_dropdowns or has_cards:
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

        # Detect prompt change — reset consumed card tracking when the prompt shifts.
        prompt_sig = f"{p_key}:{menu_title}:{player_prompt.get('promptUuid', '')}"
        if prompt_sig != getattr(self, "_last_prompt_sig", None):
            self._consumed_card_uuids.clear()
            self._last_prompt_sig = prompt_sig
        if not player_prompt or "Waiting for opponent" in menu_title:
            return

        has_buttons = "buttons" in player_prompt and len(player_prompt["buttons"]) > 0
        has_dropdowns = "dropdownListOptions" in player_prompt and len(player_prompt["dropdownListOptions"]) > 0
        selectable_uuids = player_prompt.get("selectableCards", [])
        display_cards = player_prompt.get("displayCards") or []

        is_stateful_distribution_prompt = (
            player_prompt and player_prompt.get("promptType") == "distributeAmongTargets" and player_prompt.get("distributeAmongTargets")
        )

        if has_buttons or has_dropdowns or len(selectable_uuids) > 0 or len(display_cards) > 0:
            self.active_player = p_id
            self.active_players.append(p_id)

            if has_buttons and not is_stateful_distribution_prompt:
                for btn in player_prompt["buttons"]:
                    if not btn.get("disabled", False):
                        # structured numeric features for the policy
                        features = {
                            "is_stateful": 0.0,
                            "is_macro": 0.0,
                            "is_dropdown": 1.0 if btn.get("command") == "menuButton" else 0.0,
                            "is_done": 1.0 if str(btn.get("arg", "")).lower() == "done" else 0.0,
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
                            "arg": btn.get("arg"),
                            "uuid": btn.get("uuid", ""),
                            "method": btn.get("command"),
                            "promptText": btn.get("text"),
                            "features": features,
                        })

            if has_dropdowns:
                for option in player_prompt["dropdownListOptions"]:
                    features = {
                        "is_stateful": 0.0,
                        "is_macro": 0.0,
                        "is_dropdown": 1.0,
                        "is_done": 0.0,
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

                # Add the base 'Done' button if valid.
                for btn in player_prompt.get("buttons", []):
                    if btn.get("arg") == "done" and not btn.get("disabled", False):
                        self.available_actions.append({
                            "playerId": p_id,
                            "actionType": "clickPrompt",
                            "arg": btn.get("arg"),
                            "uuid": btn.get("uuid", ""),
                            "method": btn.get("command"),
                            "promptText": btn.get("text")
                        })

                # Resourceing can only use the controlled player's hand.
                hand_cards = my_state.get("hand", [])
                if "2 cards" in menu_title:
                    for pair in itertools.combinations(hand_cards, 2):
                        features = {
                            "is_stateful": 0.0,
                            "is_macro": 1.0,
                            "is_dropdown": 0.0,
                            "is_done": 0.0,
                            "is_card": 0.0,
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
                            "uuids": [pair[0]["uuid"], pair[1]["uuid"]],
                            "arg": "any",
                            "internalName": f"{pair[0].get('internalName', 'Unknown')} + {pair[1].get('internalName', 'Unknown')}",
                            "features": features,
                        })
                else:
                    for card in hand_cards:
                        # normalize: power / 10, hp / 20
                        power_val = float(card.get("power") or card.get("printedPower") or 0.0)
                        hp_val = float(card.get("hp") or card.get("remainingHp") or card.get("currentHp") or 0.0)
                        features = {
                            "is_stateful": 0.0,
                            "is_macro": 1.0,
                            "is_dropdown": 0.0,
                            "is_done": 0.0,
                            "is_card": 0.0,
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
                            "features": features,
                        })
            elif (len(selectable_uuids) > 0 or len(display_cards) > 0) and self.current_state.get("state") and p_key in self.current_state["state"] and not is_stateful_distribution_prompt:
                my_state = self.current_state["state"][p_key]

                seen_action_uuids: set[str] = set()
                        
                # Gather only cards that can legally be selected for this prompt.
                # Keep opponent zones only for prompts that can target enemy cards.
                all_cards = {}
                for zone in ["hand", "spaceArena", "groundArena"]:
                    for card in my_state.get(zone, []):
                        all_cards[card["uuid"]] = card
                        for u in card.get("upgrades", []):
                            all_cards[u["uuid"]] = u
                if my_state.get("leader"):
                    all_cards[my_state["leader"]["uuid"]] = my_state["leader"]
                    for u in my_state["leader"].get("upgrades", []):
                        all_cards[u["uuid"]] = u
                if my_state.get("base"):
                    all_cards[my_state["base"]["uuid"]] = my_state["base"]
                    for u in my_state["base"].get("upgrades", []):
                        all_cards[u["uuid"]] = u

                opp_key = "player2" if p_key == "player1" else "player1"
                if opp_key in self.current_state["state"]:
                    opp_state = self.current_state["state"][opp_key]
                    for zone in ["spaceArena", "groundArena"]:
                        for card in opp_state.get(zone, []):
                            all_cards[card["uuid"]] = card
                            for u in card.get("upgrades", []):
                                all_cards[u["uuid"]] = u
                    if opp_state.get("leader"):
                        all_cards[opp_state["leader"]["uuid"]] = opp_state["leader"]
                        for u in opp_state["leader"].get("upgrades", []):
                            all_cards[u["uuid"]] = u
                    if opp_state.get("base"):
                        all_cards[opp_state["base"]["uuid"]] = opp_state["base"]
                        for u in opp_state["base"].get("upgrades", []):
                            all_cards[u["uuid"]] = u

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
                            "is_card": 1.0,
                            "is_friendly": 1.0 if is_friendly else 0.0,
                            "is_leader": 1.0 if is_leader else 0.0,
                            "is_base": 1.0 if is_base else 0.0,
                            "is_exhausted": 1.0 if exhausted_flag else 0.0,
                            "is_unit": 1.0 if is_unit else 0.0,
                            "card_power": power_val / 10.0,
                            "card_hp": hp_val / 20.0,
                        }

                        self.available_actions.append({
                            "playerId": p_id,
                            "actionType": "clickCard",
                            "uuid": card["uuid"],
                            "arg": "any",
                            "internalName": card.get("internalName", "Unknown"),
                            "features": features,
                        })

                # Some display-card prompts never populate selectableCards, but the visible cards are still clickable.
                # This is common for prompts that ask the player to order cards or choose from a hidden zone.
                for display_card in display_cards:
                    if not isinstance(display_card, dict):
                        continue

                    selection_state = str(display_card.get("selectionState", "")).lower()
                    if selection_state in {"viewonly", "invalid"}:
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
                        "is_card": 1.0,
                        "is_friendly": 0.0,
                        "is_leader": 0.0,
                        "is_base": 0.0,
                        "is_exhausted": 1.0 if resolved_card.get("exhausted") or resolved_card.get("isExhausted") or resolved_card.get("is_exhausted") else 0.0,
                        "is_unit": 1.0 if (resolved_card.get("power") is not None or resolved_card.get("printedPower") is not None) else 0.0,
                        "card_power": power_val / 10.0,
                        "card_hp": hp_val / 20.0,
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


    def _get_obs(self):
        """
        Extract numerical features from the JSON state into a fixed-size numpy array.
        For a CNN/Transformer, you'd map this out properly.
        Presently returns a zeroed pseudo representation.
        """
        obs = np.zeros((64,), dtype=np.float32)
        if not self.current_state or "state" not in self.current_state:
            return obs

        # Simple, deterministic feature encoder (fast to run and stable):
        # Layout (approx):
        # 0: phase hash (float)
        # 1: is_active (0/1)
        # 2-6: base/leader hp and power for controlled player
        # 7-11: base/leader hp and power for opponent
        # 12-20: counts: hand, space, ground, upgrades, resources, credits, valid_actions
        # 21-40: normalized unit totals and board advantage summary

        try:
            info = self.current_state
            phase = str(info.get("phase", "unknown"))
            # simple hash: sum of char codes modulo 100
            obs[0] = sum(ord(c) for c in phase) % 100

            # Determine which prompt/player is the active player id
            active_id = self.active_player
            my_key = None
            other_key = None
            if info.get("state"):
                # map player keys by id comparison
                p1id = info.get("player1Id")
                p2id = info.get("player2Id")
                if str(p1id) == str(self.player_id):
                    my_key = "player1"
                    other_key = "player2"
                elif str(p2id) == str(self.player_id):
                    my_key = "player2"
                    other_key = "player1"
                else:
                    my_key = "player1"
                    other_key = "player2"

            obs[1] = 1.0 if active_id and str(active_id) == str(self.player_id) else 0.0

            my_state = (info.get("state") or {}).get(my_key, {})
            other_state = (info.get("state") or {}).get(other_key, {})

            def safe_num(obj, *keys):
                for k in keys:
                    v = obj.get(k) if isinstance(obj, dict) else None
                    if v is not None:
                        try:
                            return float(v)
                        except Exception:
                            return 0.0
                return 0.0

            obs[2] = safe_num(my_state.get("base", {}), "hp", "remainingHp", "currentHp", "maxHp")
            obs[3] = safe_num(my_state.get("base", {}), "power", "printedPower")
            obs[4] = safe_num(my_state.get("leader", {}), "hp", "remainingHp", "currentHp", "maxHp")
            obs[5] = safe_num(my_state.get("leader", {}), "power", "printedPower")

            obs[6] = safe_num(other_state.get("base", {}), "hp", "remainingHp", "currentHp", "maxHp")
            obs[7] = safe_num(other_state.get("base", {}), "power", "printedPower")
            obs[8] = safe_num(other_state.get("leader", {}), "hp", "remainingHp", "currentHp", "maxHp")
            obs[9] = safe_num(other_state.get("leader", {}), "power", "printedPower")

            # counts and resources
            obs[10] = float(len(my_state.get("hand", [])))
            obs[11] = float(len(my_state.get("spaceArena", [])))
            obs[12] = float(len(my_state.get("groundArena", [])))
            obs[13] = float(my_state.get("readyResourceCount") or 0)
            obs[14] = float(my_state.get("exhaustedResourceCount") or 0)
            obs[15] = float(my_state.get("credits") or 0)
            obs[16] = float(len(self.available_actions))
            obs[17] = float(1.0 if any((a.get("actionType") == "statefulPromptResults") for a in self.available_actions) else 0.0)

            def _unit_totals(state: dict[str, Any]) -> tuple[float, float, float, float, float]:
                unit_count = 0.0
                total_power = 0.0
                total_hp = 0.0
                exhausted_count = 0.0
                leader_base_count = 0.0

                for zone in ("spaceArena", "groundArena"):
                    for c in state.get(zone, []):
                        unit_count += 1.0
                        total_power += float(safe_num(c, "power", "printedPower"))
                        total_hp += float(safe_num(c, "hp", "remainingHp", "currentHp"))
                        if c.get("exhausted") or c.get("isExhausted") or c.get("is_exhausted"):
                            exhausted_count += 1.0
                for key in ("leader", "base"):
                    card = state.get(key)
                    if isinstance(card, dict):
                        leader_base_count += 1.0
                        total_hp += float(safe_num(card, "hp", "remainingHp", "currentHp"))
                        total_power += float(safe_num(card, "power", "printedPower"))
                        if card.get("exhausted") or card.get("isExhausted") or card.get("is_exhausted"):
                            exhausted_count += 1.0

                return unit_count, total_power, total_hp, exhausted_count, leader_base_count

            my_unit_count, my_total_power, my_total_hp, my_exhausted_count, my_leader_base_count = _unit_totals(my_state)
            other_unit_count, other_total_power, other_total_hp, other_exhausted_count, other_leader_base_count = _unit_totals(other_state)

            # normalized totals / summary features
            obs[21] = my_unit_count / 10.0
            obs[22] = my_total_power / 20.0
            obs[23] = my_total_hp / 40.0
            obs[24] = my_exhausted_count / 10.0
            obs[25] = my_leader_base_count / 2.0

            obs[26] = other_unit_count / 10.0
            obs[27] = other_total_power / 20.0
            obs[28] = other_total_hp / 40.0
            obs[29] = other_exhausted_count / 10.0
            obs[30] = other_leader_base_count / 2.0

            obs[31] = (my_total_power - other_total_power) / 20.0
            obs[32] = (my_total_hp - other_total_hp) / 40.0
            obs[33] = (my_unit_count - other_unit_count) / 10.0
            obs[34] = float(my_state.get("readyResourceCount") or 0) / 8.0
            obs[35] = float(my_state.get("credits") or 0) / 8.0

        except Exception:
            # fall back to zeros on any encoding error
            pass

        return obs

    def _get_info(self):
        active_prompts = []
        if self.current_state and "prompts" in self.current_state:
            for p, pdata in self.current_state["prompts"].items():
                if pdata and "Waiting" not in pdata.get("menuTitle", ""):
                    active_prompts.append(f"{p}: '{pdata.get('menuTitle')}'")
        
        return {
            "phase": self.current_state.get("phase") if self.current_state else None,
            "activePlayer": self.active_player,
            "activePlayers": self.active_players,
            "activePrompts": active_prompts,
            "num_valid_actions": len(self.available_actions),
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
        # Prefer units with more remaining HP so we avoid overfilling low-HP units.
        candidate_cards.sort(
            key=lambda card: (
                1 if _is_unit(card) else 0,
                _remaining_hp(card) if _is_unit(card) else 0.0,
            ),
            reverse=True,
        )

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
        action_name = str(action_dict.get("actionType", "")).strip().lower()
        # Substring match: "Claim initiative", "Pass" all trigger
        full_text = f"{action_text} {action_arg}"
        is_pass_like = any(kw in full_text for kw in ("pass", "claim")) or action_name == "statefulpromptresults"

        reward = 0.0

        # Encourage actually spending available resources.
        resource_delta = (prev_ready - curr_ready) + 0.5 * (prev_credits - curr_credits)
        reward += 0.04 * resource_delta

        # Penalty for the agent passing with leftover resources OR unexhausted (ready) units.
        # Only applies when the agent is the one passing, not when the opponent ends their turn.
        action_player_id = str(action_dict.get("playerId", ""))
        is_agent_action = action_player_id == str(self.player_id)
        if is_pass_like and is_agent_action:
            # Relative waste: penalty scales with fraction of total capacity left unspent.
            # e.g. leaving 2/2 is much worse than leaving 2/6.
            prev_exhausted = float(prev_player.get("exhaustedResourceCount") or 0.0)
            total_capacity = prev_ready + prev_exhausted + 0.5 * prev_credits
            wasted = prev_ready + 0.5 * prev_credits
            waste_frac = wasted / max(1.0, total_capacity) if total_capacity > 0 else 0.0
            if waste_frac > 0.1:  # ignore <10% waste
                reward -= min(0.8, waste_frac * 1.0)
            # Count unexhausted (ready) units on the agent's board.
            curr_player_state = self._safe_state_player(current_state, curr_key)
            unexhausted = 0.0
            for zone in ("spaceArena", "groundArena"):
                for card in curr_player_state.get(zone, []):
                    if not (card.get("exhausted") or card.get("isExhausted") or card.get("is_exhausted")):
                        unexhausted += 1.0
            if unexhausted > 0:
                reward -= min(1.0, 0.2 * unexhausted)
            # Extra penalty for plain "Pass" vs "Claim initiative" — passing gives
            # the opponent a free turn without gaining initiative.
            is_claim = "claim" in full_text
            if not is_claim:
                reward -= 0.15

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
            opp_base_r = 0.25 * max(0.0, prev_opp_base - curr_opp_base)
            reward += opp_base_r

            opp_leader_r = 0.15 * max(0.0, prev_opp_leader - curr_opp_leader)
            reward += opp_leader_r

            # Losing our own base/leader is very bad.
            my_base_r = -0.25 * max(0.0, prev_my_base - curr_my_base)
            reward += my_base_r

            my_leader_r = -0.15 * max(0.0, prev_my_leader - curr_my_leader)
            reward += my_leader_r

            # Board advantage.
            board_power_delta = (curr_my_power - prev_my_power) - (curr_opp_power - prev_opp_power)
            board_power_r = 0.03 * board_power_delta
            reward += board_power_r

            board_hp_delta = (curr_my_hp - prev_my_hp) - (curr_opp_hp - prev_opp_hp)
            board_hp_r = 0.015 * board_hp_delta
            reward += board_hp_r

            # Card kills / losses.
            kill_r = 0.25 * max(0.0, prev_opp_units - curr_opp_units)
            reward += kill_r
            loss_r = -0.25 * max(0.0, prev_my_units - curr_my_units)
            reward += loss_r

        winners = current_state.get("winners", []) if current_state else []
        phase = current_state.get("phase") if current_state else None
        if len(winners) > 0 or phase == "game_end":
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
