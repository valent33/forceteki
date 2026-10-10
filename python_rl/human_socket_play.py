"""Human seat through the REAL game server, so you can watch the board in the
browser client while playing from the terminal (human-vs-human or vs agent.py).

Start the game server + client (npm run dev in forceteki and forceteki-client),
queue one seat from the browser, then run this for the other seat:

    python human_socket_play.py --deck cad_blue --decks_file decks_new.json ^
        --server_url http://localhost:9500 --player_id 111 --record human_gui.jsonl

Every action is recorded with the full GUI state (bc_train.py handles it).
"""
import argparse
import json
import random
import sys
import threading
import time

import agent as agent_mod
from distribution import remaining_hp, validate_allocation

# Card fields worth keeping for bc_train.py — drops image urls, cosmetics,
# chat, decklist blobs, etc. (a full GUI state is ~15 kB/step; this is ~1 kB).
_CARD_KEYS = ("uuid", "id", "name", "power", "hp", "remainingHp", "printedHp", "damage", "exhausted", "sentinel", "zone", "selectable")


def _slim_card(card) -> dict:
    if not isinstance(card, dict):
        return {}
    return {key: card.get(key) for key in _CARD_KEYS if key in card}


def _slim_gui_state(state: dict) -> dict:
    if not isinstance(state, dict):
        return {}
    players = state.get("players") or {}
    slim_players = {}
    for pid, ps in players.items():
        if not isinstance(ps, dict):
            continue
        piles = ps.get("cardPiles") or {}
        slim_players[pid] = {
            "id": ps.get("id"),
            "base": _slim_card(ps.get("base")),
            "leader": _slim_card(ps.get("leader")),
            # The browser serializer publishes `leaders` (a deck may run two);
            # the singular key above is kept for older recordings.
            "leaders": [_slim_card(card) for card in (ps.get("leaders") or []) if isinstance(card, dict)],
            "credits": ps.get("credits"),
            "availableResources": ps.get("availableResources"),
            "numCardsInDeck": ps.get("numCardsInDeck"),
            "promptState": ps.get("promptState"),
            "cardPiles": {
                key: [_slim_card(card) for card in (pile or []) if isinstance(card, dict)]
                for key, pile in piles.items()
                if key in ("hand", "spaceArena", "groundArena", "resources", "discard")
            },
        }
    return {"id": state.get("id"), "phase": state.get("phase"), "players": slim_players}


class HumanSocketSeat(agent_mod.QueueBotClient):
    def __init__(self, record_path: str, game_index: int, *args, **kwargs):
        kwargs.setdefault("policy_checkpoint", None)
        kwargs.setdefault("console_logging", False)
        kwargs.setdefault("verbose", False)
        super().__init__(*args, **kwargs)
        self.record_path = record_path
        self.game_index = game_index
        self._watchdog = None
        # promptUuid of the prompt we just clicked a card into (the Done/Confirm
        # button was disabled at click time). When a later push shows that
        # button enabled, we press it ourselves instead of re-asking.
        self._pending_done: str | None = None
        # promptUuids we already finished (pressed a button on). The server
        # can still deliver stale/out-of-order packets showing these prompts —
        # they must never be re-displayed.
        self._resolved_prompts: set = set()
        # promptUuids we already dumped a diagnostic for (prompt offered
        # nothing clickable — see _dump_empty_prompt).
        self._empty_prompt_dumps: set = set()

    _COLORS = {"green": "\033[32m", "cyan": "\033[36m", "yellow": "\033[33m", "red": "\033[31m", "reset": "\033[0m"}

    def _cancel_watchdog(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    def _start_watchdog(self) -> None:
        self._cancel_watchdog()
        sig = self.last_state_signature
        timer = threading.Timer(2.5, self._watchdog_fired, args=(sig,))
        timer.daemon = True
        self._watchdog = timer
        timer.start()

    def _record_action(self, state: dict, action: dict, index: int | None, auto: bool = False) -> None:
        if not self.record_path:
            return
        with open(self.record_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "game_index": self.game_index,
                "player_id": self.player_id,
                "state": _slim_gui_state(state),
                "chosen_action": action,
                "action_index": index,
                "auto": auto,
            }) + "\n")

    def _emit_auto_done(self, state: dict, prompt_state: dict, prompt_uuid: str) -> None:
        button = next(
            (candidate for candidate in (prompt_state.get("buttons") or [])
             if not candidate.get("disabled")
             and str(candidate.get("arg", "")).strip().lower() == "done"
             and "skip" not in str(candidate.get("text", "")).strip().lower()),
            None
        )
        text = (button or {}).get("text", "Done")
        action = {
            "kind": (button or {}).get("command") or "menuButton",
            "actionType": "clickPrompt",
            "arg": (button or {}).get("arg", "done"),
            "uuid": prompt_uuid,
            "method": (button or {}).get("command") or "menuButton",
            "description": f"auto {text}",
        }
        self._emit_action(action)
        self._last_emit = (action, 1)
        self.last_action_description = action.get("description")
        self.last_action_kind = action.get("kind")
        self.last_prompt_uuid = prompt_uuid
        self.steps_taken += 1
        self._start_watchdog()
        self._record_action(state, action, None, auto=True)
        print(f"  (auto: {text})")

    def _is_safe_retry(self, action) -> bool:
        # Buttons/stateful answers are idempotent-ish; card clicks are NOT
        # (re-clicking in a multi-select prompt toggles the selection off).
        # Exception: when our card click was never acknowledged by the server
        # (we're still pending the done press), it was dropped — re-send once.
        action_type = action.get("actionType")
        if action_type == "clickCard":
            return self._pending_done is not None
        if action_type in {"clickPrompt", "menuButton"}:
            arg = str(action.get("arg") or "")
            prompt = self._current_prompt()
            button_args = {str(b.get("arg") or "") for b in (prompt.get("buttons") or []) if isinstance(b, dict)}
            dropdowns = {str(option) for option in (prompt.get("dropdownListOptions") or [])}
            if arg not in button_args and arg not in dropdowns:
                # A display-card selection is sent as menuButton(<card uuid>).
                # Re-sending it would toggle the card back OFF.
                return False
            return True
        return action_type == "statefulPromptResults"

    def _watchdog_fired(self, sig) -> None:
        with self.action_lock:
            if self.game_over or self.last_state_signature != sig:
                return
            last_emit = getattr(self, "_last_emit", None)
            if last_emit is not None and last_emit[1] < 2 and self._is_safe_retry(last_emit[0]):
                # The server never answered — likely a dropped click during the
                # other seat's turn. Re-send the same button press once.
                self._last_emit = (last_emit[0], last_emit[1] + 1)
                self._start_watchdog()
                self._emit_action(last_emit[0])
                return
            # Otherwise: re-display the prompt.
            self.last_state_signature = None
        print("  (no visible change — re-picking the current prompt)")
        self._maybe_take_action()

    def _done_enabled(self, prompt_state: dict) -> bool:
        return any(
            not button.get("disabled")
            and str(button.get("arg", "")).strip().lower() == "done"
            and "skip" not in str(button.get("text", "")).strip().lower()
            for button in (prompt_state.get("buttons") or [])
        )

    def _state_sig(self, state: dict, prompt_state: dict) -> tuple:
        # Noise-immune but content-aware. Cosmetic churn (playerIsNewlyActive
        # flipping, a button's text changing) must not re-display the prompt —
        # that was doubling every prompt. But a changed prompt *content* MUST
        # re-display it: e.g. a "put cards on top in any order" prompt shrinks
        # its displayCards after each per-card click, and ignoring that made the
        # client think its click did nothing and stop prompting.
        selectable = tuple(sorted(
            str(card.get("uuid"))
            for card in self._collect_selectable_cards(state)
            if card.get("uuid")
        ))
        selected = tuple(sorted(
            str(card.get("uuid"))
            for card in self._collect_selectable_cards(state)
            if card.get("selected")
        ))
        display = tuple(
            (str(card.get("cardUuid") or card.get("uuid") or ""), str(card.get("selectionState") or ""))
            for card in (prompt_state.get("displayCards") or [])
            if isinstance(card, dict)
        )
        per_card_buttons = tuple(
            str(button.get("arg") or button.get("text") or "")
            for button in (prompt_state.get("perCardButtons") or [])
            if isinstance(button, dict)
        )
        buttons = tuple(
            (str(button.get("text") or ""), str(button.get("arg") or ""), bool(button.get("disabled")))
            for button in (prompt_state.get("buttons") or [])
            if isinstance(button, dict)
        )
        return (
            prompt_state.get("promptUuid"),
            prompt_state.get("promptType"),
            prompt_state.get("menuTitle"),
            selectable,
            selected,
            display,
            per_card_buttons,
            buttons,
        )

    def _maybe_take_action(self) -> None:
        with self.action_lock:
            if self.game_over:
                return
            state = self.current_state
            if not state or "players" not in state:
                return
            player_state = state["players"].get(str(self.player_id))
            if not player_state:
                return
            prompt_state = player_state.get("promptState") or {}
            prompt_uuid = prompt_state.get("promptUuid")
            if not prompt_uuid:
                return
            # Stale packet: this prompt is already finished server-side.
            if prompt_uuid in self._resolved_prompts:
                return
            state_signature = self._state_sig(state, prompt_state)
            if state_signature == self.last_state_signature:
                return
            self.last_state_signature = state_signature
            self._cancel_watchdog()

            # Auto-done: our previous card click landed and the Done/Confirm
            # button just became enabled — press it ourselves. Re-asking here
            # is what caused the double-click (a second click UNSELECTS the
            # card) and the "skipped resourcing" reports.
            if self._pending_done == prompt_uuid:
                if self._done_enabled(prompt_state):
                    self._pending_done = None
                    self._resolved_prompts.add(prompt_uuid)
                    self._emit_auto_done(state, prompt_state, prompt_uuid)
                    return
                # Selection registered but Done is still disabled (multi-card
                # prompt): keep pending and fall through so the human can pick
                # the next card (already-selected cards are excluded).
            else:
                self._pending_done = None

            action = self._choose_action(state, prompt_state)
            if not action:
                if not self.game_over and not self._build_candidates(state, prompt_state):
                    self._dump_empty_prompt(state, prompt_state)
                return
            self._emit_action(action)
            self._last_emit = (action, 1)
            self.last_action_description = action.get("description", "unknown")
            self.last_action_kind = action.get("kind")
            self.last_prompt_uuid = prompt_uuid
            self.steps_taken += 1
            self._start_watchdog()
            if action.get("actionType") in {"clickCard", "statefulPromptResults"}:
                if self._done_enabled(prompt_state):
                    # Done is already enabled (e.g. a view-cards prompt): the
                    # click is its own answer — never auto-press anything.
                    self._pending_done = None
                else:
                    self._pending_done = prompt_uuid
            else:
                # A button press can resolve the prompt — mark it resolved so
                # stale pushes never re-display it.
                self._resolved_prompts.add(prompt_uuid)

    def _log_prompt_state(self, player_state, prompt_state) -> None:
        pass  # we print our own candidates

    def _card_index(self, state: dict) -> dict[str, tuple[str, str, dict]]:
        """uuid -> (owner, zone, card) across both players (leaders included)."""
        return super()._card_index(state)

    @staticmethod
    def _prompt_card_uuids(state: dict, prompt_state: dict) -> set:
        """Every card uuid the prompt currently offers (display + selectable)."""
        uuids: set[str] = set()
        for card in (prompt_state.get("displayCards") or []):
            if isinstance(card, dict):
                uuid = str(card.get("cardUuid") or card.get("uuid") or "")
                if uuid:
                    uuids.add(uuid)
        for card in (state.get("players") or {}).values():
            if not isinstance(card, dict):
                continue
            for pile in (card.get("cardPiles") or {}).values():
                for entry in (pile or []):
                    if isinstance(entry, dict) and entry.get("selectable") and entry.get("uuid"):
                        uuids.add(str(entry["uuid"]))
            for entry in agent_mod.iter_leader_cards(card):
                if entry.get("selectable"):
                    uuids.add(str(entry["uuid"]))
            base = card.get("base")
            if isinstance(base, dict) and base.get("selectable") and base.get("uuid"):
                uuids.add(str(base["uuid"]))
        return uuids

    def _dump_empty_prompt(self, state: dict, prompt_state: dict) -> None:
        """One-shot diagnostic when a prompt offers us nothing clickable."""
        prompt_uuid = str(prompt_state.get("promptUuid") or "")
        if prompt_uuid in self._empty_prompt_dumps:
            return
        self._empty_prompt_dumps.add(prompt_uuid)
        print(f"  (no clickable option for prompt {prompt_state.get('menuTitle')!r} "
              f"type={prompt_state.get('promptType')!r})")
        try:
            with open("prompt_debug_gui.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "prompt": prompt_state,
                    "state": _slim_gui_state(state),
                }) + "\n")
            print("  (prompt state dumped to prompt_debug_gui.jsonl)")
        except Exception as exc:
            print(f"  (could not dump prompt state: {exc})")

    def _candidate_label(self, state: dict, candidate: dict) -> str:
        """Card candidates get a [zone] / (opp) / EX tag so two identical
        cards (e.g. one in hand, one in play) are tellable apart. Stateful
        distribution candidates show what they will actually do."""
        base = f"{candidate.get('kind')} {candidate.get('description', '?')}"
        result = candidate.get("result") or {}
        distribution = result.get("valueDistribution") or []
        if distribution:
            index = self._card_index(state)
            bits = []
            for entry in distribution:
                owner, zone, card = index.get(str(entry.get("uuid")), (None, None, {}))
                name = card.get("name") or entry.get("uuid")
                bits.append(f"{entry.get('amount')} -> {name}")
            return f"{base} ({', '.join(bits)})"
        uuid = candidate.get("cardUuid") or candidate.get("uuid")
        if not uuid:
            return base
        owner, zone, card_info = self._card_index(state).get(str(uuid), (None, None, None))
        if zone is None:
            return base
        tag = f" [{zone}]"
        if owner == "opp":
            tag += " (opp)"
        if card_info and card_info.get("exhausted"):
            tag += " EX"
        return base + tag

    def _print_board(self, state: dict) -> None:
        players = state.get("players") or {}
        ps = players.get(str(self.player_id)) or {}
        piles = ps.get("cardPiles") or {}
        db = agent_mod._card_db_by_id()

        def card_line(card, with_cost=False):
            if not isinstance(card, dict):
                return None
            name = card.get("name") or card.get("id") or "?"
            db_card = db.get(str(card.get("id"))) or {}
            cost = db_card.get("cost")
            mark = " EX" if card.get("exhausted") else ""
            stats = ""
            if card.get("power") is not None or card.get("hp") is not None:
                stats = f" {card.get('power')}/{card.get('hp')}"
            if with_cost and cost is not None:
                return f"{name}(c{cost}){stats}{mark}"
            return f"{name}{stats}{mark}"

        base = ps.get("base") or {}
        print(f"  hand: {[card_line(c, with_cost=True) for c in (piles.get('hand') or []) if card_line(c, with_cost=True)]}")
        print(f"  base: {base.get('name') or '?'} {base.get('hp')} | "
              f"resources: {ps.get('availableResources', '?')} ready | credits: {ps.get('credits', '?')} | "
              f"deck: {ps.get('numCardsInDeck', '?')}")
        # `leaders` (array) is what the browser serializer sends; older
        # recordings and the RL env use a singular `leader`.
        for leader in agent_mod.iter_leader_cards(ps):
            line = card_line(leader)
            if line:
                print(f"  leader: {line}" + ("  [clickable]" if leader.get("selectable") else ""))
        for arena in ("groundArena", "spaceArena"):
            mine = [card_line(c) for c in (piles.get(arena) or []) if card_line(c)]
            print(f"  {arena}: {mine}")

    def _pick_action(self, state: dict, candidates: list, prompt_state: dict) -> int | None:
        """Returns a candidate index, or None for refresh/quit."""
        title = prompt_state.get("menuTitle") or prompt_state.get("promptTitle") or "Choose"
        labels = [self._candidate_label(state, candidate) for candidate in candidates]
        try:
            import questionary  # optional: pip install questionary for arrow-key menus
        except Exception:
            questionary = None
        if questionary is not None:
            choice = questionary.select(title, choices=labels).ask()# + ["(quit)"]
            if choice in (None, "(quit)"):
                self.game_over = True
                return None
            return labels.index(choice)
        print("\n" + "=" * 78)
        print(f"[{self.player_id}] {title}")
        for index, label in enumerate(labels):
            print(f"  [{index}] {label}")
        print("=" * 78)
        while True:
            raw = input("action> ").strip().lower()
            if raw in {"q", "quit", "exit"}:
                self.game_over = True
                return None
            if raw in {"r", "refresh", ""}:
                return None
            try:
                index = int(raw)
            except ValueError:
                print("  invalid input — number, r, or q")
                continue
            if not 0 <= index < len(candidates):
                print(f"  index out of range (0..{len(candidates) - 1})")
                continue
            return index

    @staticmethod
    def _prompt_sig(prompt_state: dict) -> tuple:
        return (
            prompt_state.get("promptUuid"),
            len(prompt_state.get("selectedCards") or []),
            prompt_state.get("menuTitle"),
        )

    def _current_prompt(self) -> dict:
        current = self.current_state or {}
        player_state = (current.get("players") or {}).get(str(self.player_id)) or {}
        return player_state.get("promptState") or {}

    # ── distribute among targets ─────────────────────────────────────────────

    def _distribution_targets(self, state, prompt_state):
        """[(card, is_friendly)] for the legal targets the SERVER marked."""
        index = self._card_index(state)
        uuids = [str(uuid) for uuid in (prompt_state.get("selectableCards") or []) if uuid]
        entries = []
        seen: set = set()
        if uuids:
            for uuid in uuids:
                found = index.get(uuid)
                if found and uuid not in seen:
                    seen.add(uuid)
                    entries.append((found[2], found[0] == "you"))
        else:
            for card in self._collect_selectable_cards(state):
                uuid = str(card.get("uuid") or "")
                if not uuid or uuid in seen:
                    continue
                seen.add(uuid)
                found = index.get(uuid)
                entries.append((card, (found[0] == "you") if found else True))
        return entries

    @staticmethod
    def _distribution_action(prompt_state, distribution_type, distribution):
        return {
            "kind": "statefulPromptResults",
            "actionType": "statefulPromptResults",
            "uuid": prompt_state.get("promptUuid"),
            "result": {"type": distribution_type, "valueDistribution": distribution},
            "description": f"statefulPromptResults {distribution_type}",
        }

    @staticmethod
    def _parse_allocation(raw: str, num_targets: int):
        parts = raw.replace(",", " ").replace(":", "=").split()
        if not parts:
            return None
        if all("=" not in part for part in parts):
            try:
                values = [int(part) for part in parts]
            except ValueError:
                return None
            if len(values) > num_targets:
                return None
            return {i: value for i, value in enumerate(values)}
        allocation: dict = {}
        for part in parts:
            if "=" not in part:
                return None
            left, _, right = part.partition("=")
            try:
                index, value = int(left), int(right)
            except ValueError:
                return None
            if not 0 <= index < num_targets:
                return None
            allocation[index] = value
        return allocation

    @staticmethod
    def _even_allocation(entry_count: int, amount: int, max_targets) -> dict:
        limit = entry_count
        if max_targets:
            limit = min(limit, int(max_targets))
        allocation: dict = {}
        for step in range(amount):
            allocation[step % limit] = allocation.get(step % limit, 0) + 1
        return allocation

    def _choose_distribution(self, state, prompt_state):
        """Interactive allocation for a distribute-among-targets prompt."""
        prompt = prompt_state.get("distributeAmongTargets") or {}
        dtype = str(prompt.get("type") or "")
        amount = int(prompt.get("amount") or 0)
        can_less = bool(prompt.get("canDistributeLess"))
        can_none = bool(prompt.get("canChooseNoTargets"))
        max_targets = int(prompt["maxTargets"]) if prompt.get("maxTargets") else None
        kind = {
            "distributeDamage": "damage",
            "distributeIndirectDamage": "indirect damage",
            "distributeHealing": "healing",
            "distributeTokenUpgrade": "token upgrades",
        }.get(dtype, dtype)

        entries = self._distribution_targets(state, prompt_state)
        if not entries:
            if can_none:
                print("  (no legal targets — distributing nothing)")
                return self._distribution_action(prompt_state, dtype, [])
            print("  (no legal targets for this distribution)")
            return None
        cards_by_uuid = {str(card.get("uuid")): card for card, _friendly in entries}

        def label(index_: int) -> str:
            card, friendly = entries[index_]
            hp = remaining_hp(card)
            hp_text = f" {hp}hp" if hp is not None else ""
            return f"{card.get('name') or card.get('uuid')}{hp_text} ({'yours' if friendly else 'opp'})"

        print(f"\n  Distribute {amount} {kind}"
              + (f" to up to {max_targets} target(s)" if max_targets else "")
              + (" — may distribute less" if can_less else "")
              + (" — may choose nothing" if can_none else ""))
        for i in range(len(entries)):
            print(f"   [{i}] {label(i)}")
        print("  amounts like `1=2 3=2`, a bare list `2 2`, or `even` / `all` / `none` / `done`")

        allocation: dict = {}
        while True:
            try:
                raw = input("  distribute> ").strip().lower()
            except EOFError:
                return None
            if raw in {"q", "quit", "exit"}:
                self.game_over = True
                return None
            if raw in {"r", "refresh"}:
                return None
            if raw in {"even", "spread"}:
                allocation = self._even_allocation(len(entries), amount, max_targets)
            elif raw == "all":
                allocation = {0: amount}
            elif raw in {"none", "no"}:
                if not can_none:
                    print("  you must distribute something")
                    continue
                allocation = {}
            elif raw not in {"", "done", "d"}:
                parsed = self._parse_allocation(raw, len(entries))
                if parsed is None:
                    print("  couldn't parse that — try `1=2 3=2`, `2 2`, `even`, `all`, `none`")
                    continue
                allocation = parsed

            by_uuid = {str(entries[i][0].get("uuid")): int(value) for i, value in allocation.items()}
            error = validate_allocation(prompt, by_uuid, cards_by_uuid)
            if error:
                current = " ".join(f"{i}={v}" for i, v in sorted(allocation.items()) if v) or "nothing"
                print(f"  invalid: {error} (current: {current})")
                continue
            break

        distribution = [
            {"uuid": str(entries[i][0].get("uuid")), "amount": int(value)}
            for i, value in sorted(allocation.items()) if int(value) > 0
        ]
        summary = ", ".join(f"{value} -> {label(i)}" for i, value in sorted(allocation.items()) if value) or "nothing"
        print(f"  distributing: {summary}")
        return self._distribution_action(prompt_state, dtype, distribution)

    def _choose_action(self, state, prompt_state):
        self._print_board(state)
        if prompt_state.get("promptType") == "distributeAmongTargets" and prompt_state.get("distributeAmongTargets"):
            action = self._choose_distribution(state, prompt_state)
            if action is None:
                return None
            self._record_action(state, action, None)
            return action
        candidates = self._build_candidates(state, prompt_state)
        if not candidates:
            return None
        # Dedupe (e.g. the GUI can list the opponent's leader both in a pile
        # and as the `leader` field — same card, one option).
        seen: set = set()
        unique: list = []
        for candidate in candidates:
            key = (candidate.get("kind"), candidate.get("cardUuid") or candidate.get("arg") or candidate.get("description"))
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        candidates = unique
        index = self._pick_action(state, candidates, prompt_state)
        if index is None:
            return None
        # Debounce: the other seat may be acting; wait for the state to settle
        # (no new push for ~0.45s) so the click doesn't fire into a stale
        # prompt that the server then silently drops.
        stable_sig = None
        deadline = time.time() + 1.5
        while time.time() < deadline:
            sig = self._prompt_sig(self._current_prompt())
            if sig == stable_sig:
                break
            stable_sig = sig
            time.sleep(0.15)
        cur_prompt = self._current_prompt()
        if (self._prompt_sig(cur_prompt) != self._prompt_sig(prompt_state)
                or self._state_sig(self.current_state or state, cur_prompt) != self._state_sig(state, prompt_state)):
            print("  prompt changed while choosing — re-picking…")
            return self._choose_action(self.current_state or state, cur_prompt)

        # Hold the click until the engine's current pipeline step belongs to us.
        # Clicks sent while the other player's prompt is on top are silently
        # dropped — this is the "resource on the GUI first and the CLI goes
        # crazy" bug.
        waited_hint = False
        hold_deadline = time.time() + 15.0
        wanted_uuid = prompt_state.get("promptUuid")
        while True:
            state_now = self.current_state or {}
            owner = state_now.get("activePromptPlayerId")
            current_uuid = self._current_prompt().get("promptUuid")
            if owner is None or str(owner) == str(self.player_id):
                break
            if wanted_uuid and current_uuid and str(current_uuid) != str(wanted_uuid):
                # The prompt we answered is no longer on screen — don't hold
                # for a stale one.
                break
            if not waited_hint:
                print("  (holding — waiting for the opponent to finish their action)")
                waited_hint = True
            if time.time() > hold_deadline:
                # Emitting is safer than dropping: the server ignores illegal
                # clicks, and the watchdog re-prompts if nothing changes.
                print("  (waited 15s for the opponent — sending the action anyway)")
                break
            time.sleep(0.2)

        action = candidates[index]
        # Stale-prompt guard: display cards are re-dealt and can become
        # invalid/unselectable between the menu being drawn and the click
        # landing. Clicking a card the server no longer has in the prompt is
        # silently rejected (no state change), which looks like a freeze.
        target = str(action.get("cardUuid") or action.get("arg") or "")
        if target and target in self._prompt_card_uuids(state, prompt_state):
            fresh_state = self.current_state or state
            if target not in self._prompt_card_uuids(fresh_state, self._current_prompt()):
                print("  that card is no longer part of the prompt — re-picking…")
                return self._choose_action(fresh_state, self._current_prompt())
        self._record_action(state, action, index)
        return action


def main() -> int:
    parser = argparse.ArgumentParser(description="Human seat on the real game server (GUI-spectated)")
    parser.add_argument("--server_url", default="http://localhost:9500")
    parser.add_argument("--deck", default="", help="Deck key for this seat (ignored with --random_decks)")
    parser.add_argument("--decks_file", default="decks_new.json")
    parser.add_argument("--random_decks", action="store_true", help="Sample a random deck from the decks file each game")
    parser.add_argument("--player_id", default="111", help="Id to queue with (must not clash with the other seat)")
    parser.add_argument("--record", default="human_gui.jsonl", help="JSONL path to append actions to")
    parser.add_argument("--games", type=int, default=1, help="Number of consecutive games")
    parser.add_argument("--yes", action="store_true", help="Auto-restart queueing without prompts")
    args = parser.parse_args()

    keys = []
    if args.random_decks:
        with open(args.decks_file, "r", encoding="utf-8") as handle:
            keys = sorted(json.load(handle).keys())
    if not args.random_decks and not args.deck:
        parser.error("--deck or --random_decks required")

    for game in range(args.games):
        deck = random.choice(keys) if args.random_decks else args.deck
        print(f"Game {game + 1}: queuing seat {args.player_id} with deck {deck}")
        client = HumanSocketSeat(
            record_path=args.record,
            game_index=game,
            server_url=args.server_url,
            deck_key=deck,
            decks_file=args.decks_file,
            player_id=args.player_id,
            max_steps=2000,
            display_name=f"human-{args.player_id}",
        )
        client.connect_and_queue()
        client.wait()
        print(f"\nGame {game + 1} finished (seat {args.player_id}).")
        if game < args.games - 1:
            if not args.yes and input("Next game? [enter/q] ").strip().lower() in {"q", "quit"}:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
