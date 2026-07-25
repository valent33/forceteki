import json
import os


def load_deck(deck_key, file_path="decks.json"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Deck file {file_path} not found. Please create it with a JSON dictionary mapping deck keys to deck objects.")

    with open(file_path, "r", encoding="utf-8") as f:
        decks_db = json.load(f)

    if deck_key not in decks_db:
        raise KeyError(f"Deck key '{deck_key}' not found in {file_path}")

    data = decks_db[deck_key]
    deck_list = []

    # Load SWUDB ID -> internalName mappings dynamically
    mapping = {}
    try:
        # Resolve paths relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        set_code_path = os.path.join(base_dir, "../test/json/_setCodeMap.json")
        card_map_path = os.path.join(base_dir, "../test/json/_cardMap.json")

        if os.path.exists(set_code_path) and os.path.exists(card_map_path):
            with open(set_code_path, "r", encoding="utf-8") as f_set:
                set_codes = json.load(f_set)
            with open(card_map_path, "r", encoding="utf-8") as f_card:
                card_db = json.load(f_card)

            # Map number IDs to internal names
            num_to_internal = {str(c["id"]): c["internalName"] for c in card_db if "id" in c and "internalName" in c}

            # Map set codes (like SEC_006) to internal names
            for code, num_id in set_codes.items():
                if str(num_id) in num_to_internal:
                    mapping[code] = num_to_internal[str(num_id)]
                elif isinstance(num_id, list):
                    for n_id in num_id:
                        if str(n_id) in num_to_internal:
                            mapping[code] = num_to_internal[str(n_id)]
                            break
    except Exception as e:
        print(f"Warning: Failed to load dynamic deck mappings: {e}")

    def _resolve_id(raw_id):
        return mapping.get(raw_id, raw_id)

    def _append_card(raw_id, count):
        resolved_id = _resolve_id(raw_id)
        # if resolved_id == "underworld-thug":
        #     print(f"[deck-load] {deck_key}: {raw_id} -> underworld-thug x{count}")
        # elif resolved_id != raw_id:
        #     print(f"[deck-load] {deck_key}: {raw_id} -> {resolved_id} x{count}")

        for _ in range(count):
            deck_list.append(resolved_id)

    if isinstance(data.get("deck"), dict):
        for card_name, count in data["deck"].items():
            _append_card(card_name, count)
    elif isinstance(data.get("cards"), list):
        for card_obj in data["cards"]:
            count = card_obj.get("count", 1)
            _append_card(card_obj["id"], count)
    elif isinstance(data.get("deck"), list):
        for card_obj in data["deck"]:
            count = card_obj.get("count", 1)
            _append_card(card_obj["id"], count)

    leader = data.get("leader", {}).get("id") if isinstance(data.get("leader"), dict) else data.get("leader", "darth-vader#dark-lord-of-the-sith")
    base = data.get("base", {}).get("id") if isinstance(data.get("base"), dict) else data.get("base", "kestro-city")

    if len(deck_list) < 50:
        raise ValueError(f"Deck '{deck_key}' contains {len(deck_list)} cards; expected at least 50.")

    return _resolve_id(leader), _resolve_id(base), deck_list
