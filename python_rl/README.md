# python_rl

Quick reminders for the Python RL scripts in this folder.

## Prereqs

Run these from the repo root with the project venv active.

```powershell
.\.venv\Scripts\Activate.ps1
```

If your environment is missing dependencies, the scripts expect at least `torch`, `numpy`, `requests`, and `gymnasium`.

## Server

Either run:
```
node build/server/rl/envServer.js
```
for headless or:
```
npm run dev
```
for GUI.


## Train a policy

`train.py` runs RL training against the server and writes checkpoints/logs under `python_rl/runs/train` by default.

```powershell
python .\python_rl\train.py --p1 vader --p2 luke --episodes 100
```

Useful flags:

- `--randomize_decks` sample fresh decks from `python_rl/decks.json` each episode.
- `--checkpoint <path>` resume from a saved `.pt` or `.ckpt` file.
- `--checkpoint_every N` write numbered checkpoints every `N` episodes.
- `--server_url http://localhost:3005` override the game server address.

## Play headless bot vs bot

`test_agent.py` is the simplest way to run two deck selections against the RL environment server.

```powershell
python .\python_rl\test_agent.py --p1 vader --p2 luke
```

Useful flags:

- `--server_url http://localhost:3005` point at a different RL server.
- `--player_id 111` choose which side the agent controls.
- `--log_dir <path>` save run logs somewhere else.
- `--verbose` print board state and actions every turn.

## Play with the GUI

`agent.py` is the single-seat policy agent for the GUI client. By default it loads `runs/train_run/policy_latest.ckpt`.

```powershell
python .\python_rl\agent.py --deck vader
```

Useful flags:

- `--policy_checkpoint <path>` choose a different checkpoint.
- `--player_id 111` or `--player_id 222` pick the side you control.
- `--verbose` print board state and action details.

If you want the agent to create the game on the server first, use reset mode:

```powershell
python .\python_rl\agent.py --deck vader --reset --opponent_deck luke --server_url http://localhost:3005
```

Also run:
```
npm run dev
```
in forceteki-client\ to run the Gweb GUI

## Notes

- Training and the headless runner both assume the Forceteki server is already running.
- GUI mode assumes the browser client is open and connected.
- Outputs from recent runs live under `python_rl/runs/`.

# forceteki workflow

Update upstream:
git fetch upstream
git checkout main
git pull upstream main

Update feature branch:
git checkout custom-rl-engine
git rebase main

After rebase:
git push --force-with-lease
