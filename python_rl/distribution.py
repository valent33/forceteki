"""Shared "distribute among targets" logic for the RL and GUI clients.

Three call sites used to each re-implement this (``agent.py``, ``swu_env.py``,
``human_socket_play.py``) and all of them dumped the entire amount onto a single
target. That is usually wrong (Advantage/Experience tokens want spreading) and
sometimes illegal (indirect damage must not exceed a unit's remaining HP).

The server tells us the legal targets (card-level ``selectable`` / prompt
``selectableCards``) and the prompt parameters; this module turns that into a
valid, sensible ``valueDistribution``.

``allocate_distribution(prompt_data, targets)`` where ``targets`` is an iterable
of ``(card, friendly_bool)`` returns a list of ``{"uuid", "amount"}`` or ``None``
when no legal allocation exists (in which case leave the prompt unanswered).
"""
from __future__ import annotations

from typing import Any, Iterable

STATE_DAMAGE = "distributeDamage"
STATE_INDIRECT = "distributeIndirectDamage"
STATE_HEALING = "distributeHealing"
STATE_TOKEN = "distributeTokenUpgrade"

DAMAGE_TYPES = (STATE_DAMAGE, STATE_INDIRECT)


def remaining_hp(card: dict[str, Any]) -> int | None:
    """Best-effort remaining HP across every state serializer we consume."""
    if not isinstance(card, dict):
        return None

    for key in ("remainingHp", "currentHp"):
        value = card.get(key)
        if value is not None:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                pass

    try:
        hp = int(card.get("hp"))
    except (TypeError, ValueError):
        return None

    damage = card.get("damage")
    if damage is not None:
        try:
            return max(0, hp - int(damage))
        except (TypeError, ValueError):
            pass
    return max(0, hp)


def is_unit(card: dict[str, Any]) -> bool:
    return isinstance(card, dict) and card.get("power") is not None


def _rank_key(prompt_type: str, card: dict[str, Any], friendly: bool, amount: int):
    hp = remaining_hp(card)
    hp_key = hp if hp is not None else 99
    try:
        power = float(card.get("power") or 0.0)
    except (TypeError, ValueError):
        power = 0.0
    try:
        damage = float(card.get("damage") or 0.0)
    except (TypeError, ValueError):
        damage = 0.0
    friendly_key = 0 if friendly else 1

    if prompt_type in DAMAGE_TYPES:
        # Focus fire on opponents: finish off what we can kill, then the
        # weakest remaining unit, then the biggest threat.
        lethal = 0 if hp is not None and hp <= amount else 1
        return (friendly_key, lethal, hp_key, -power)
    if prompt_type == STATE_HEALING:
        # Heal the most damaged friendlies first.
        return (friendly_key, -damage, hp_key)
    if prompt_type == STATE_TOKEN:
        # Tokens want spreading across friendly units; best attackers first.
        return (friendly_key, 0 if is_unit(card) else 1, -power, -hp_key)
    return (friendly_key,)


def _cap(prompt_type: str, card: dict[str, Any]) -> int | None:
    """Per-target maximum (None = unlimited)."""
    if prompt_type == STATE_INDIRECT:
        # Indirect damage may never exceed a unit's remaining HP (server contract).
        hp = remaining_hp(card)
        return hp if hp is not None else 0
    if prompt_type == STATE_HEALING:
        # Prefer not to overheal, but the server clamps it, so this cap is a
        # preference: allocate_distribution tops up past it rather than failing.
        hp = remaining_hp(card)
        printed = card.get("printedHp")
        if hp is None or printed is None:
            return None
        try:
            return max(0, int(printed) - hp)
        except (TypeError, ValueError):
            return None
    return None


def allocate_distribution(
    prompt_data: dict[str, Any],
    targets: Iterable[tuple[dict[str, Any], bool]],
) -> list[dict[str, Any]] | None:
    """Return a valid ``valueDistribution`` for a distribute prompt, or None.

    ``targets`` yields ``(card, is_friendly)``; card dicts need at least a
    ``uuid`` (remaining HP/power/damage make the allocation smarter).
    """
    prompt_type = str(prompt_data.get("type") or "")
    try:
        amount = int(prompt_data.get("amount") or 0)
    except (TypeError, ValueError):
        amount = 0
    if not prompt_type or amount <= 0:
        return None

    can_less = bool(prompt_data.get("canDistributeLess"))
    can_none = bool(prompt_data.get("canChooseNoTargets"))
    max_targets = prompt_data.get("maxTargets")
    try:
        max_targets = int(max_targets) if max_targets else None
    except (TypeError, ValueError):
        max_targets = None

    entries: list[tuple[dict[str, Any], bool]] = [
        (card, bool(friendly))
        for card, friendly in targets
        if isinstance(card, dict) and card.get("uuid")
    ]
    if not entries:
        return [] if can_none else None

    entries.sort(key=lambda item: _rank_key(prompt_type, item[0], item[1], amount))
    spread = prompt_type == STATE_TOKEN

    allocation: dict[str, int] = {}
    remaining = amount
    if spread:
        # One point at a time, best target first, so tokens get spread out.
        while remaining > 0:
            progressed = False
            for card, _friendly in entries:
                if remaining <= 0:
                    break
                uuid = str(card["uuid"])
                limit = _cap(prompt_type, card)
                current = allocation.get(uuid, 0)
                if limit is not None and current >= limit:
                    continue
                if uuid not in allocation and max_targets is not None and len(allocation) >= max_targets:
                    continue
                allocation[uuid] = current + 1
                remaining -= 1
                progressed = True
            if not progressed:
                break
    else:
        for card, _friendly in entries:
            if remaining <= 0:
                break
            uuid = str(card["uuid"])
            if uuid not in allocation and max_targets is not None and len(allocation) >= max_targets:
                break
            limit = _cap(prompt_type, card)
            current = allocation.get(uuid, 0)
            if limit is not None and current >= limit:
                continue
            step = remaining if limit is None else min(remaining, limit - current)
            if step <= 0:
                continue
            allocation[uuid] = current + step
            remaining -= step

    if remaining > 0:
        if prompt_type == STATE_INDIRECT:
            # Nowhere left to put it without exceeding remaining HP.
            return None
        if not can_less:
            # Damage/tokens have no per-target cap — top up the best target.
            first = str(entries[0][0]["uuid"])
            allocation[first] = allocation.get(first, 0) + remaining
            remaining = 0

    if not allocation:
        return [] if can_none else None
    if not can_less and sum(allocation.values()) != amount:
        return None
    if max_targets is not None and len(allocation) > max_targets:
        return None
    if prompt_type == STATE_INDIRECT:
        # Only indirect damage has a hard per-target cap.
        for card, _friendly in entries:
            uuid = str(card["uuid"])
            limit = _cap(prompt_type, card)
            if uuid in allocation and limit is not None and allocation[uuid] > limit:
                return None

    return [{"uuid": uuid, "amount": value} for uuid, value in allocation.items()]


def validate_allocation(
    prompt_data: dict[str, Any],
    allocation: dict[str, int],
    cards_by_uuid: dict[str, dict[str, Any]],
) -> str | None:
    """Check a human-entered allocation; returns an error string or None."""
    prompt_type = str(prompt_data.get("type") or "")
    try:
        amount = int(prompt_data.get("amount") or 0)
    except (TypeError, ValueError):
        amount = 0
    can_less = bool(prompt_data.get("canDistributeLess"))
    can_none = bool(prompt_data.get("canChooseNoTargets"))
    max_targets = prompt_data.get("maxTargets")
    try:
        max_targets = int(max_targets) if max_targets else None
    except (TypeError, ValueError):
        max_targets = None

    amounts = {uuid: int(value) for uuid, value in allocation.items() if int(value) > 0}
    total = sum(amounts.values())

    if any(int(value) < 0 for value in allocation.values()):
        return "amounts cannot be negative"
    if total == 0:
        return None if can_none else "you must distribute at least 1"
    if can_less:
        if total > amount:
            return f"you can distribute at most {amount} (currently {total})"
    elif total != amount:
        return f"you must distribute exactly {amount} (currently {total})"
    if max_targets is not None and len(amounts) > max_targets:
        return f"at most {max_targets} target(s) allowed, you used {len(amounts)}"
    if prompt_type == STATE_INDIRECT:
        for uuid, value in amounts.items():
            card = cards_by_uuid.get(uuid) or {}
            limit = _cap(prompt_type, card)
            if limit is not None and value > limit:
                name = card.get("name") or card.get("internalName") or uuid
                return f"{name} only has {limit} HP left"
    return None
