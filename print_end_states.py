"""Extract end states from trajectory CSVs and write them to a single log file.

Usage:
    python BALROG/print_end_states.py <trajectory_dir> [-o OUTPUT_FILE]

Example:
    python BALROG/print_end_states.py BALROG/logs/dev/feb6/2026-02-06_13-58-16_robust_cot_google_gemini-2.5-flash_perc/minihack/MiniHack-Quest-Easy-v0
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def extract_end_state(csv_path: Path) -> dict:
    """Parse a trajectory CSV and return the last row (end state)."""
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        last_row = None
        for row in reader:
            last_row = row
    return last_row


def load_run_metadata(json_path: Path) -> dict | None:
    """Load summary metadata from the companion JSON file."""
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Extract end states from trajectory CSVs")
    parser.add_argument("trajectory_dir", type=Path, help="Directory containing trajectory CSV/JSON files")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output log file (default: end_states.log in trajectory_dir)")
    args = parser.parse_args()

    traj_dir = args.trajectory_dir
    if not traj_dir.is_dir():
        print(f"Error: {traj_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(traj_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: no CSV files found in {traj_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or traj_dir / "end_states.log"

    with open(output_path, "w") as out:
        for csv_path in csv_files:
            run_name = csv_path.stem
            json_path = csv_path.with_suffix(".json")

            out.write(f"{'=' * 80}\n")
            out.write(f"RUN: {run_name}\n")
            out.write(f"{'=' * 80}\n")

            # Load metadata from JSON
            meta = load_run_metadata(json_path)
            if meta:
                out.write(f"  Steps:          {meta.get('num_steps', 'N/A')}\n")
                out.write(f"  Episode Return: {meta.get('episode_return', 'N/A')}\n")
                out.write(f"  Progression:    {meta.get('progression', 'N/A')}\n")
                out.write(f"  End Reason:     {meta.get('end_reason', 'N/A')}\n")
                out.write(f"  Done:           {meta.get('done', 'N/A')}\n")
                out.write(f"  Seed:           {meta.get('seed', 'N/A')}\n")
                out.write(f"  Cost:           ${meta.get('total_cost', 'N/A')}\n")
                actions = meta.get("action_frequency", {})
                if actions:
                    out.write(f"  Actions:        {dict(actions)}\n")
                out.write("\n")

            # Extract end state from CSV
            end_row = extract_end_state(csv_path)
            if end_row is None:
                out.write("  [No rows found in CSV]\n\n")
                continue

            out.write(f"  Final Step:   {end_row.get('Step', 'N/A')}\n")
            out.write(f"  Final Action: {end_row.get('Action', 'N/A')}\n")
            out.write(f"  Final Reward: {end_row.get('Reward', 'N/A')}\n")
            out.write(f"  Done:         {end_row.get('Done', 'N/A')}\n")

            reasoning = end_row.get("Reasoning", "").strip()
            if reasoning:
                out.write(f"\n  --- Final Reasoning ---\n")
                for line in reasoning.splitlines():
                    out.write(f"  {line}\n")

            observation = end_row.get("Observation", "").strip()
            if observation:
                out.write(f"\n  --- Final Observation ---\n")
                for line in observation.splitlines():
                    out.write(f"  {line}\n")

            out.write("\n\n")

    print(f"Wrote end states for {len(csv_files)} runs to {output_path}")


if __name__ == "__main__":
    main()
