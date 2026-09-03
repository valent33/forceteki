"""Human-vs-human play + action recording against ONE env server.

One terminal per seat. The host (seat 111) creates the games and picks decks;
the guest (seat 222) joins whatever game is live. Each terminal writes its own
JSONL of its own actions:

    {game_index, player_id, state, chosen_action, action_index}

Start an env server first (fresh process per game; the host resets it between
games — reset is re-entrant on one process):

    node build/server/rl/envServer.js 3120

Terminal A (host):
    python human_play.py --player_id 111 --server_url http://localhost:3120 \\
        --decks_file decks_new.json --record human_seat111.jsonl

Terminal B (guest):
    python human_play.py --player_id 222 --server_url http://localhost:3120 \\
        --decks_file decks_new.json --record human_seat222.jsonl

Controls per turn: type the action index, `r` to refresh, `q` to quit.
"""
import argparse
import copy
import json
import random
import sys
import time

from deck_utils import load_deck
from swu_env import SWUEnv


def _deck_keys(decks_file: str) -> list[str]:
    with open(decks_file, "r", encoding="utf-8") as handle:
        db = json.load(handle)
    if not isinstance(db, dict):
        raise ValueError(f"Expected {decks_file} to contain a JSON object of deck definitions")
    return sorted(str(key) for key in db.keys())


def _build_reset_payload(p1_key, p2_key, decks_file: str) -> dict:
    reset_options: dict = {"phase": "setup", "player1": {"hasInitiative": True}}
    if p1_key:
        leader, base, deck = load_deck(p1_key, decks_file)
        reset_options["p1Leader"] = leader
        reset_options["p1Base"] = base
        reset_options["p1Cards"] = deck
    if p2_key:
        leader, base, deck = load_deck(p2_key, decks_file)
        reset_options["p2Leader"] = leader
        reset_options["p2Base"] = base
        reset_options["p2Cards"] = deck
    return reset_options


def _winner_from_state(state: dict, player_id: str) -> str | None:
    section = state.get("state") or {}
    agent_key = "player1" if str(state.get("player1Id")) == str(player_id) else "player2"
    opp_key = "player2" if agent_key == "player1" else "player1"

    def hp(key: str) -> float:
        base = (section.get(key) or {}).get("base") or {}
        return float(base.get("hp") or base.get("remainingHp") or base.get("currentHp") or base.get("maxHp") or 0.0)

    my_hp, opp_hp = hp(agent_key), hp(opp_key)
    if my_hp <= 0 and opp_hp > 0:
        return "opponent"
    if opp_hp <= 0 and my_hp > 0:
        return "you"
    if my_hp <= 0 and opp_hp <= 0:
        return "draw"
    return None


def _record(record_path: str, game_index: int, player_id: str, state: dict, action: dict, action_index: int) -> None:
    if not record_path:
        return
    line = {
        "game_index": game_index,
        "player_id": player_id,
        "state": state,
        "chosen_action": action,
        "action_index": action_index,
    }
    with open(record_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")


def _print_board(env: SWUEnv, player_id: str) -> None:
    """Compact board view: hand with costs, bases, leader, arenas."""
    state = env.current_state or {}
    section = state.get("state") or {}
    my_key = "player1" if str(state.get("player1Id")) == str(player_id) else "player2"
    opp_key = "player2" if my_key == "player1" else "player1"
    me = section.get(my_key) or {}
    opp = section.get(opp_key) or {}

    def hand_line(card):
        if not isinstance(card, dict):
            return None
        db = env._card_data(card.get("internalName"))
        cost = db.get("cost")
        suffix = f" (cost {cost})" if cost is not None else ""
        return f"{card.get('internalName')}{suffix}"

    base = me.get("base") or {}
    opp_base = opp.get("base") or {}
    leader = me.get("leader") or {}
    lines = [
        f"  hand: {[hand_line(c) for c in me.get('hand', []) if hand_line(c)]}",
        f"  base: {base.get('hp', '?')} | opp base: {opp_base.get('hp', '?')} | "
        f"resources: {me.get('readyResourceCount', 0)} ready / {me.get('exhaustedResourceCount', 0)} exhausted | "
        f"credits: {me.get('credits', 0)}",
    ]
    if leader:
        lines.append(
            f"  leader: {leader.get('internalName')} zone={leader.get('zone')} "
            f"exhausted={leader.get('exhausted')} hp={leader.get('hp')}"
        )
    for arena in ("groundArena", "spaceArena"):
        def unit_line(card):
            if not isinstance(card, dict):
                return None
            mark = " EX" if card.get("exhausted") else ""
            return f"{card.get('internalName')} {card.get('power')}/{card.get('hp')}{mark}"
        mine = [unit_line(c) for c in me.get(arena, []) if unit_line(c)]
        theirs = [unit_line(c) for c in opp.get(arena, []) if unit_line(c)]
        lines.append(f"  {arena}: ME {mine} | OPP {theirs}")
    print("\n".join(lines))


def _play_game(env: SWUEnv, player_id: str, game_index: int, record_path: str, reset_factory) -> bool:
    """Play one game for this seat. `reset_factory` is None for the guest."""
    print(f"\n===== Game {game_index + 1} — you are seat {player_id} =====")
    if reset_factory is not None:
        payload, p1_key, p2_key = reset_factory()
        print(f"  decks: {p1_key} vs {p2_key}")
        try:
            env.reset(options=payload)
        except Exception as exc:
            print(f"  reset failed: {type(exc).__name__}: {exc}")
            return False

    last_wait_msg = None
    while True:
        try:
            env.refresh()
        except Exception as exc:
            print(f"  refresh failed: {type(exc).__name__}: {exc}")
            return False

        state = env.current_state or {}
        if "error" in state:
            print("  server returned an error state — waiting…")
            time.sleep(0.5)
            continue

        prompts = state.get("prompts") or {}
        my_key = "player1" if str(state.get("player1Id")) == str(player_id) else "player2"
        my_prompt = prompts.get(my_key) or {}
        title = my_prompt.get("menuTitle", "")

        if not title or "waiting for opponent" in title.lower():
            msg = f"  … waiting ({title or 'no prompt yet'})"
            if msg != last_wait_msg:
                print(msg)
                last_wait_msg = msg
            time.sleep(0.4)
            continue
        last_wait_msg = None

        print("\n" + "=" * 78)
        print(f"[{player_id}] phase={state.get('phase')} | {title}")
        _print_board(env, player_id)
        actions = list(env.available_actions)
        if not actions:
            print("  no actions available — refreshing…")
            time.sleep(0.4)
            continue
        for index, action in enumerate(actions):
            label = action.get("internalName") or action.get("promptText") or action.get("arg") or "?"
            print(f"  [{index}] {action.get('actionType')} {label}")
        print("=" * 78)

        raw = input("action> ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            return False
        if raw in {"", "r", "refresh"}:
            continue
        try:
            index = int(raw)
        except ValueError:
            print("  invalid input — number, r, or q")
            continue
        if not 0 <= index < len(actions):
            print(f"  index out of range (0..{len(actions) - 1})")
            continue

        action = copy.deepcopy(actions[index])
        state_before = copy.deepcopy(env.current_state)
        try:
            _, reward, terminated, truncated, _ = env.step(index)
        except Exception as exc:
            print(f"  step failed: {type(exc).__name__}: {exc}")
            continue

        _record(record_path, game_index, player_id, state_before, action, index)
        print(f"  -> {action.get('actionType')} "
              f"{action.get('internalName') or action.get('promptText') or action.get('arg') or '?'} "
              f"(reward={reward:.2f})")

        if terminated or truncated:
            winner = _winner_from_state(env.current_state or {}, player_id)
            print(f"\n  GAME OVER — winner: {winner or 'unresolved'}")
            return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Human seat controller + recorder for the RL env server")
    parser.add_argument("--player_id", default="111", choices=["111", "222"], help="Which seat you control (111 = host)")
    parser.add_argument("--server_url", default="http://localhost:3120")
    parser.add_argument("--decks_file", default="decks_new.json")
    parser.add_argument("--deck", help="Deck key for seat 111 (host only)")
    parser.add_argument("--opponent_deck", help="Deck key for seat 222 (host only)")
    parser.add_argument("--record", default="", help="JSONL path to append this seat's actions to")
    args = parser.parse_args()

    env = SWUEnv(server_url=args.server_url, player_id=args.player_id, human_mode=True)
    keys = _deck_keys(args.decks_file)
    is_host = args.player_id == "111"
    game_index = 0

    def reset_factory():
        p1_key = args.deck or random.choice(keys)
        p2_key = args.opponent_deck or random.choice(keys)
        return _build_reset_payload(p1_key, p2_key, args.decks_file), p1_key, p2_key

    while True:
        if is_host:
            answer = input(f"\nStart game {game_index + 1}? [enter=random decks / q=quit] ").strip().lower()
            if answer in {"q", "quit"}:
                break
        else:
            # Guest: wait until the host's game appears on the server.
            print(f"\nWaiting for the host to start game {game_index + 1}…")
            while True:
                try:
                    env.refresh()
                except Exception:
                    time.sleep(0.5)
                    continue
                state = env.current_state or {}
                if "error" not in state and state.get("phase") == "setup":
                    break
                time.sleep(0.5)

        keep_going = _play_game(env, args.player_id, game_index, args.record, reset_factory if is_host else None)
        game_index += 1
        if not keep_going:
            break

    print(f"\nSession over — recorded {game_index} game(s) for seat {args.player_id}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
