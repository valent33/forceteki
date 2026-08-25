"""Inspect the raw payloads the RL env server sends at every game state.

The env server (server/rl/envServer.ts) is the legality authority: it sends
`selectableCards` (engine-filtered legal click targets), `buttons` (with
`disabled` flags and args), `dropdownListOptions`, `displayCards`,
`perCardButtons`, and `debug_legalActions` (per hand card: which actions meet
their requirements and which are play actions).

Usage:
  1. Start the env server (forceteki root):   node build/server/rl/envServer.js
  2. Run:                                     python inspect_server.py
  3. At each step, the full server payload is printed. Paste a raw /step JSON
     body (one line) and press Enter to advance the game, e.g.:
       {"playerId":"111","action":"clickPrompt","arg":"keep","promptText":"Keep"}
     Type 'quit' to stop.

Optional args:  python inspect_server.py [server_url]
"""

import json
import sys

import requests

DEFAULT_SERVER = "http://localhost:3005"


def fmt_card(card: dict | None) -> str:
    if not card:
        return "?"
    uuid = str(card.get("uuid", ""))
    return f"{card.get('internalName', '?')}({uuid[:8]})"


def dump_prompt(state: dict, seat: str) -> None:
    prompt = (state.get("prompts") or {}).get(seat) or {}
    title = str(prompt.get("menuTitle", ""))
    if not title or "waiting for opponent" in title.lower():
        return
    print(f"\n=== phase={state.get('phase')} activePlayer={state.get('activePlayer')} seat={seat} ===")
    print(f"menuTitle: {title!r}")
    print(f"promptType: {str(prompt.get('promptType'))!r}  promptUuid: {prompt.get('promptUuid')}")
    buttons = prompt.get("buttons") or []
    print(f"buttons ({len(buttons)}):")
    for b in buttons:
        print(f"  text={str(b.get('text'))!r} arg={str(b.get('arg'))!r} disabled={b.get('disabled')} command={b.get('command')}")
    dropdowns = prompt.get("dropdownListOptions") or []
    print(f"dropdownListOptions ({len(dropdowns)}): {dropdowns[:12]}")
    cards = (state.get("state") or {}).get(seat) or {}
    all_cards: dict[str, dict] = {}
    for zone in ("hand", "spaceArena", "groundArena", "discard", "deck", "resources"):
        for c in cards.get(zone) or []:
            all_cards[str(c.get("uuid"))] = c
    for special in ("leader", "base"):
        c = cards.get(special)
        if c:
            all_cards[str(c.get("uuid"))] = c
    sels = prompt.get("selectableCards") or []
    print(f"selectableCards ({len(sels)}):")
    for uuid in sels:
        print(f"  {fmt_card(all_cards.get(str(uuid)))}  uuid={uuid}")
    print(f"selectedCards: {prompt.get('selectedCards')}")
    print("debug_legalActions (hand cards):")
    for entry in prompt.get("debug_legalActions") or []:
        for a in entry.get("actions") or []:
            print(f"  {entry.get('id')}: title={a.get('title')} req={a.get('req')} isPlay={a.get('isPlay')}")
    per_card = prompt.get("perCardButtons") or []
    if per_card:
        print(f"perCardButtons: {per_card}")


def main() -> int:
    server = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SERVER
    reset_payload = {"options": {"phase": "action", "player1": {"hasInitiative": True}}}
    resp = requests.post(f"{server}/reset", json=reset_payload, timeout=15)
    resp.raise_for_status()
    state = resp.json()
    print("Game reset. Paste raw /step JSON bodies to advance; 'quit' to stop.")
    while True:
        dump_prompt(state, "player1")
        dump_prompt(state, "player2")
        line = input("\naction (raw /step JSON, or 'quit'): ").strip()
        if line.lower() in {"quit", "exit", "q"}:
            break
        try:
            body = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"bad json: {exc}")
            continue
        resp = requests.post(f"{server}/step", json=body, timeout=15)
        if resp.status_code != 200:
            print(f"step failed {resp.status_code}: {resp.text[:300]}")
        state = resp.json()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
