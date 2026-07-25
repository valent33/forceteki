import json
import os

# Update these paths if your logs are in a different subdirectory
INPUT_FILE = "runs/train_run/transitions.jsonl"
SUMMARY_ONLY_FILE = "runs/train_run/episode_summaries.jsonl"
TRANSITIONS_ONLY_FILE = "runs/train_run/transitions_fixed.jsonl"

def untangle():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"Reading {INPUT_FILE} (this might take a minute for 3GB)...")
    
    total_count = 0
    summary_count = 0
    transition_count = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(SUMMARY_ONLY_FILE, 'w', encoding='utf-8') as f_sum, \
         open(TRANSITIONS_ONLY_FILE, 'w', encoding='utf-8') as f_trans:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            try:
                # We parse each line to check the event type
                data = json.loads(line)
                
                if data.get("event") == "episode_summary":
                    f_sum.write(line + "\n")
                    summary_count += 1
                else:
                    f_trans.write(line + "\n")
                    transition_count += 1
            except json.JSONDecodeError:
                continue

            total_count += 1
            if total_count % 5000 == 0:
                print(f"Processed {total_count} lines... Found {summary_count} summaries.", end='\r')

    print(f"\n\nProcessing Complete!")
    print(f"---------------------------------")
    print(f"Total lines: {total_count}")
    print(f"Summaries extracted: {summary_count} -> {SUMMARY_ONLY_FILE}")
    print(f"Clean transitions: {transition_count} -> {TRANSITIONS_ONLY_FILE}")

if __name__ == "__main__":
    untangle()