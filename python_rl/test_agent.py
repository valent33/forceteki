import argparse
import json
from swu_env import SWUEnv
from policy import RandomActionPolicy
from runner import EpisodeLogger, SingleAgentEpisodeRunner
from deck_utils import load_deck

def main():
    parser = argparse.ArgumentParser(description="SWU Headless RL Agent")
    parser.add_argument("--decks_file", type=str, default="decks.json", help="JSON file containing the deck dictionaries")
    parser.add_argument("--p1", type=str, help="Deck key for Player 1 (e.g., 'vader')")
    parser.add_argument("--p2", type=str, help="Deck key for Player 2 (e.g., 'luke')")
    parser.add_argument("--server_url", type=str, default="http://localhost:3005", help="RL env server URL")
    parser.add_argument("--player_id", type=str, default="111", help="Player id for the controlled side")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum number of steps before stopping")
    parser.add_argument("--log_dir", type=str, help="Optional directory for logs")
    parser.add_argument("--verbose", action="store_true", help="Print board and actions every turn")
    args = parser.parse_args()

    print(f"Connecting to SWU Headless RL Server at {args.server_url}...")
    env = SWUEnv(server_url=args.server_url, player_id=args.player_id, single_agent_mode=True)

    reset_options = {
        "phase": "setup",
        "player1": {"hasInitiative": True}
    }
    
    if args.p1:
        leader, base, deck = load_deck(args.p1, args.decks_file)
        reset_options["p1Leader"] = leader
        reset_options["p1Base"] = base
        reset_options["p1Cards"] = deck

    if args.p2:
        leader, base, deck = load_deck(args.p2, args.decks_file)
        reset_options["p2Leader"] = leader
        reset_options["p2Base"] = base
        reset_options["p2Cards"] = deck
        
    reset_payload = {
        "p1Leader": reset_options.get("p1Leader"),
        "p1Base": reset_options.get("p1Base"),
        "p1Cards": reset_options.get("p1Cards"),
        "p2Leader": reset_options.get("p2Leader"),
        "p2Base": reset_options.get("p2Base"),
        "p2Cards": reset_options.get("p2Cards"),
        "options": {
            "phase": "setup",
            "player1": {"hasInitiative": True}
        }
    }
    logger = EpisodeLogger(log_dir=args.log_dir, verbose=args.verbose)
    runner = SingleAgentEpisodeRunner(
        env=env,
        policy=RandomActionPolicy(),
        logger=logger,
        player_id=args.player_id,
        max_steps=args.max_steps,
    )

    try:
        runner.run(reset_options=reset_payload)
    finally:
        logger.close()

    print(f"\n--- Simulation Complete in {runner.steps} steps ---")

if __name__ == "__main__":
    main()