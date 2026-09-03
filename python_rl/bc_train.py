"""Behavior-cloning warm start from recorded human games (human_play.py JSONL).

Rebuilds the exact action list the human chose among from each recorded state,
then fine-tunes the Q(s,a) head with cross-entropy on the human's actions.

    python bc_train.py --data human_seat111.jsonl human_seat222.jsonl \
        --checkpoint runs/train_2/policy_latest.ckpt --out policy_bc.ckpt \
        --epochs 8 --lr 3e-4

Without --checkpoint the policy starts from random init (pure BC).
"""
import argparse
import json
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

import train as train_mod
from swu_env import SWUEnv
from torch_policy import ACTION_FEATURE_DIM, _action_features, TorchPolicy


def _match_action_index(actions: list[dict], action: dict) -> int | None:
    a_type = action.get("actionType")
    a_uuid = str(action.get("uuid") or action.get("cardUuid") or "")
    a_arg = str(action.get("arg") or "")
    a_text = str(action.get("promptText") or action.get("description") or "")

    if a_type is None:
        return None
    for i, cand in enumerate(actions):
        if cand.get("actionType") != a_type:
            continue
        if a_uuid and str(cand.get("uuid") or cand.get("cardUuid") or "") == a_uuid:
            return i
    for i, cand in enumerate(actions):
        if cand.get("actionType") != a_type:
            continue
        if a_arg and str(cand.get("arg") or "") == a_arg:
            return i
    for i, cand in enumerate(actions):
        if cand.get("actionType") != a_type:
            continue
        if a_text and str(cand.get("promptText") or cand.get("description") or "") == a_text:
            return i
    return None


def _load_samples(data_paths: list[str], env: SWUEnv, max_samples: int) -> tuple[list, list, list, list, int, int]:
    obs_list: list[np.ndarray] = []
    feat_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    target_list: list[int] = []
    matched = skipped = 0

    for path in data_paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or "state" not in record or "chosen_action" not in record:
                    skipped += 1
                    continue
                if max_samples and len(obs_list) >= max_samples:
                    break
                player_id = str(record.get("player_id") or "111")
                if env.player_id != player_id:
                    env.player_id = player_id
                raw_state = record["state"]
                if isinstance(raw_state, dict) and "players" in raw_state and "state" not in raw_state:
                    # GUI-socket format (recorded by human_socket_play.py):
                    # convert to the env-server shape first.
                    from agent import QueueBotClient
                    converter = QueueBotClient.__new__(QueueBotClient)
                    converter.player_id = player_id
                    raw_state = converter._gui_state_to_env_state(raw_state)
                env.current_state = raw_state
                try:
                    env._update_available_actions()
                except Exception:
                    skipped += 1
                    continue
                actions = list(env.available_actions[:env.max_action_space])
                index = _match_action_index(actions, record["chosen_action"])
                if index is None:
                    skipped += 1
                    continue
                obs_list.append(env._get_obs().astype(np.float32))
                feat_list.append(np.asarray([_action_features(a) for a in actions], dtype=np.float32))
                mask_list.append(np.asarray(env.legal_action_mask, dtype=np.int8).copy())
                target_list.append(int(index))
                matched += 1
            if max_samples and len(obs_list) >= max_samples:
                break

    return obs_list, feat_list, mask_list, target_list, matched, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Behavior-cloning warm start from recorded human games")
    parser.add_argument("--data", nargs="+", required=True, help="human_play.py JSONL files")
    parser.add_argument("--checkpoint", default=None, help="Existing RL checkpoint to warm start from (optional)")
    parser.add_argument("--out", default="policy_bc.ckpt", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=0, help="Cap on samples (0 = all)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env = SWUEnv(server_url="http://localhost:9", player_id="111", single_agent_mode=True)
    policy = TorchPolicy(obs_size=int(env.observation_space.shape[0]), max_actions=100, lr=args.lr, device=args.device)
    if args.checkpoint:
        train_mod._load_checkpoint(policy, args.checkpoint, args.device)
        print(f"Loaded checkpoint from {args.checkpoint}")

    obs_list, feat_list, mask_list, target_list, matched, skipped = _load_samples(
        args.data, env, args.max_samples
    )
    total = matched + skipped
    print(f"BC dataset: {matched} matched samples, {skipped} skipped"
          + (f" (of {total} records)" if total else ""))
    if not obs_list:
        print("No usable samples — check that the JSONL files came from human_play.py")
        return 1

    indices = list(range(len(obs_list)))
    obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(args.device)

    for epoch in range(1, args.epochs + 1):
        random.shuffle(indices)
        epoch_loss = 0.0
        epoch_correct = 0
        batches = 0
        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            batch_obs = obs_tensor[batch_idx]
            batch_feats = [torch.tensor(feat_list[i], dtype=torch.float32).to(args.device) for i in batch_idx]
            batch_masks = [mask_list[i] for i in batch_idx]
            batch_targets = torch.tensor([target_list[i] for i in batch_idx], dtype=torch.long).to(args.device)

            q_logits, _ = policy.evaluate_q(batch_obs, batch_feats, batch_masks)
            loss = F.cross_entropy(q_logits, batch_targets)
            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()

            epoch_loss += float(loss.item()) * len(batch_idx)
            epoch_correct += int((q_logits.argmax(dim=1) == batch_targets).sum().item())
            batches += 1

        epoch_loss /= max(1, len(indices))
        acc = epoch_correct / max(1, len(indices))
        print(f"epoch {epoch}/{args.epochs}: loss={epoch_loss:.4f} accuracy={acc:.1%}")

    payload = {
        "model_state_dict": {key: value.detach().cpu() for key, value in policy.net.state_dict().items()},
        "optimizer_state_dict": policy.optimizer.state_dict(),
        "episode": 0,
        "obs_size": policy.obs_size,
        "max_actions": policy.max_actions,
        "checkpoint_source": args.checkpoint,
        "bc_meta": {"samples": matched, "skipped": skipped, "epochs": args.epochs, "lr": args.lr},
    }
    torch.save(payload, args.out)
    print(f"Saved BC checkpoint to {args.out} (final accuracy {acc:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
