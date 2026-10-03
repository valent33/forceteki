"""Q-head end-to-end diagnostic.

Run this against TWO freshly started env servers (each server handles exactly
ONE game per process, so restart them before every run):

    node build/server/rl/envServer.js 3110
    node build/server/rl/envServer.js 3111

Then:

    python qhead_diag.py                      # uses 3110 (trace) + 3111 (real run)
    python qhead_diag.py 3101 3102            # custom ports

What it checks:
  1. A fully traced manual episode against server A — every step prints
     active player, available actions, the chosen action, reward and
     termination, and any raised exception gets a full traceback.
  2. The real train._run_worker_episode against server B, including the
     main-process async-A2C Q-head recompute (evaluate_q + logsumexp) and
     one policy.update.
"""
import sys
import traceback

import numpy as np
import requests
import torch

import train
from swu_env import SWUEnv
from torch_policy import ACTION_FEATURE_DIM, TorchPolicy
from policy import RandomActionPolicy

DECK_A = "yularen-droid"
DECK_B = "yularen-droid"
PLAYER_ID = "111"
MAX_STEPS = 40
TRACE_STEPS = 20


def _label(action) -> str:
    return (
        action.get("internalName")
        or action.get("promptText")
        or action.get("arg")
        or action.get("actionType")
        or "?"
    )


def traced_episode(url: str) -> None:
    print(f"\n=== TRACED EPISODE against {url} ===")
    env = SWUEnv(server_url=url, player_id=PLAYER_ID, single_agent_mode=True)
    policy = TorchPolicy(obs_size=2386, max_actions=100, device="cpu", temperature=1.5)
    opponent = RandomActionPolicy()
    payload = train._build_reset_payload(DECK_A, DECK_B, "decks.json")[0]

    try:
        _, info = env.reset(options=payload)
    except Exception as exc:
        print(f"RESET FAILED ({type(exc).__name__}):")
        traceback.print_exc()
        return

    terminated = False
    for step in range(1, MAX_STEPS + 1):
        active = str(info.get("activePlayer") or "")
        is_agent = active == str(PLAYER_ID)
        actor = policy if is_agent else opponent
        print(f"\n-- step {step}  active={active or '?'}  "
              f"activePlayers={info.get('activePlayers')}  "
              f"agent_turn={is_agent}  phase={info.get('phase')}")

        if not env.available_actions:
            print(f"   available_actions EMPTY -> refresh()")
            try:
                _, info = env.refresh()
            except Exception:
                print("   refresh FAILED:")
                traceback.print_exc()
                break
            continue

        n = len(env.available_actions)
        legal = int(env.legal_action_mask.sum())
        print(f"   actions={n} legal_mask={legal}/100")
        for i, a in enumerate(env.available_actions[:8]):
            mark = "M" if not bool(env.legal_action_mask[i]) else " "
            print(f"     [{i}]{mark} {a.get('actionType')} {_label(a)}")

        obs_vec = torch.tensor(env._get_obs(), dtype=torch.float32)
        logp = None
        try:
            action, logp, value = actor.select_action(
                obs_vec, list(env.available_actions), env.legal_action_mask
            )
        except AttributeError:
            # Same fallback as train._run_worker_episode for legacy policies.
            action = actor.choose_action_index(env)
        except Exception:
            print("   select_action FAILED:")
            traceback.print_exc()
            break
        if action is None:
            print("   select_action -> None")
            try:
                _, info = env.refresh()
            except Exception:
                print("   refresh FAILED:")
                traceback.print_exc()
                break
            continue

        print(f"   chose [{action}] {env.available_actions[action].get('actionType')} "
              f"{_label(env.available_actions[action])}")
        try:
            _, reward, terminated, truncated, info = env.step(action)
        except Exception:
            print("   STEP FAILED (this is what silently zeroes the episode):")
            traceback.print_exc()
            break
        print(f"   -> reward={reward} terminated={terminated} truncated={truncated}")
        if terminated or truncated:
            print(f"   episode over at step {step} (terminated={terminated})")
            break
        if step >= TRACE_STEPS:
            print(f"   (trace cap {TRACE_STEPS} reached, stopping trace)")
            break

    state = env.current_state or {}
    section = state.get("state") or {}
    for seat in ("player1", "player2"):
        base = (section.get(seat) or {}).get("base") or {}
        print(f"   END {seat} base hp={base.get('hp', '?')}")


def _server_reachable(url: str) -> bool:
    # Before the first /reset the server answers /state with a 500
    # {"error":"Game not initialized"} — any HTTP response means it's up.
    try:
        requests.get(f"{url}/state", timeout=10)
        return True
    except requests.exceptions.ConnectionError as exc:
        print(f"   unreachable: {type(exc).__name__}: {exc}")
        return False
    except Exception as exc:
        # Proxy/other transport issues: the server may still be fine, but we
        # can't talk to it — report what happened.
        print(f"   unreachable: {type(exc).__name__}: {exc}")
        return False


def real_worker_episode(url: str) -> bool:
    print(f"\n=== REAL WORKER EPISODE against {url} ===")
    env = SWUEnv(server_url=url, player_id=PLAYER_ID, single_agent_mode=True)
    policy = TorchPolicy(obs_size=2386, max_actions=100, device="cpu", temperature=1.5)
    opponent = RandomActionPolicy()
    payload = train._build_reset_payload(DECK_A, DECK_B, "decks.json")[0]
    res = train._run_worker_episode(env, policy, opponent, PLAYER_ID, payload, MAX_STEPS)
    if res is None:
        print("FAIL: _run_worker_episode returned None (reset raised).")
        # Re-raise the reset once so the real error is visible.
        try:
            env.reset(options=payload)
        except Exception as exc:
            print(f"   reset error: {type(exc).__name__}: {exc}")
        return False
    obs_list, actions, feat_list, mask_list, rewards, steps, winner, ahp, ohp, *_deck_keys = res
    print(f"   steps={steps} samples={len(obs_list)} winner={winner} "
          f"agent_hp={ahp} opp_hp={ohp}")
    if not obs_list:
        print("FAIL: zero samples collected. See the traced episode above for why.")
        return False
    if not (len(obs_list) == len(actions) == len(feat_list) == len(mask_list) == len(rewards)):
        print("FAIL: sample list lengths differ.")
        return False
    for f in feat_list:
        if f.shape[1] != ACTION_FEATURE_DIM:
            print(f"FAIL: feature tensor shape {tuple(f.shape)}.")
            return False

    obs_batch = torch.tensor(np.stack(obs_list), dtype=torch.float32)
    feat_tensors = [torch.tensor(f, dtype=torch.float32) for f in feat_list]
    q_logits, values = policy.evaluate_q(obs_batch, feat_tensors, mask_list)
    chosen = torch.tensor(actions, dtype=torch.long)
    logps = q_logits.gather(1, chosen.unsqueeze(1)).squeeze(1) - torch.logsumexp(q_logits, dim=1)
    assert logps.requires_grad and torch.isfinite(logps).all(), "bad logps"
    ret = train.discounted_returns(rewards, gamma=0.99)
    tl, pl, vl, ent = policy.update(list(logps), ret, list(values), value_coef=0.5, entropy_coef=0.1)
    print(f"   evaluate_q+update OK: loss={tl:.4f} policy={pl:.4f} value={vl:.4f} entropy={ent:.3f}")
    return True


if __name__ == "__main__":
    url_a = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3110"
    url_b = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:3111"
    print("=== SERVER CHECK ===")
    a_ok = _server_reachable(url_a)
    b_ok = _server_reachable(url_b)
    print(f"   {url_a}: {'reachable' if a_ok else 'NOT REACHABLE'}")
    print(f"   {url_b}: {'reachable' if b_ok else 'NOT REACHABLE'}")
    if not (a_ok and b_ok):
        print("\n=== RESULT: FAIL (one or both env servers are not up;")
        print("    start BOTH fresh servers before running this — each server")
        print("    only accepts ONE game per process) ===")
        sys.exit(1)
    traced_episode(url_a)
    ok = real_worker_episode(url_b)
    print("\n=== RESULT:", "PASS" if ok else "FAIL", "===")
    sys.exit(0 if ok else 1)
