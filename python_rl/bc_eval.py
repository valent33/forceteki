import argparse, random, numpy as np, torch
import train as train_mod, bc_train
from swu_env import SWUEnv
from torch_policy import TorchPolicy

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--data", nargs="+", required=True)
p.add_argument("--val_split", type=float, default=0.15)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--batch_size", type=int, default=64)
a = p.parse_args()

env = SWUEnv(server_url="http://localhost:9", player_id="111", single_agent_mode=True)
policy = TorchPolicy(obs_size=int(env.observation_space.shape[0]), max_actions=100, device="cpu")
train_mod._load_checkpoint(policy, a.checkpoint, "cpu")

obs, feats, masks, targets, matched, skipped = bc_train._load_samples(a.data, env, 0)
idx = list(range(len(obs)))
random.Random(a.seed).shuffle(idx)
val = idx[: max(1, int(len(idx) * a.val_split))]

obs_t = torch.tensor(np.stack(obs), dtype=torch.float32)
correct = 0
with torch.no_grad():
    for s in range(0, len(val), a.batch_size):
        b = val[s:s + a.batch_size]
        q, _ = policy.evaluate_q(
            obs_t[b],
            [torch.tensor(feats[i], dtype=torch.float32) for i in b],
            [masks[i] for i in b],
        )
        correct += int((q.argmax(1) == torch.tensor([targets[i] for i in b])).sum())
print(f"{a.checkpoint}: val accuracy {correct / len(val):.1%} ({len(val)} samples)")
