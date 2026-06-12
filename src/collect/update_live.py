"""
update_live.py
--------------
Lit les scores depuis scores.txt et met à jour la simulation.

Format scores.txt (un match par ligne) :
  2026-06-11 | A | Mexico | South Africa | 2-0
  2026-06-11 | A | South Korea | Czech Republic | 2-1

Usage :
  python src/collect/update_live.py
"""

import pandas as pd
import re
import sys
import subprocess
from pathlib import Path
from loguru import logger

SCORES_FILE = Path("scores.txt")
OUTPUT_FILE = Path("data/raw/historical/wc2026_results_live.csv")
SIMULATE_SCRIPT = Path("src/model/simulate.py")

import io
logger.remove()
logger.add(io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8"), 
           format="{time:HH:mm:ss} | {message}")
def parse_scores() -> pd.DataFrame:
    if not SCORES_FILE.exists():
        logger.error(f"{SCORES_FILE} not found! Create it first.")
        sys.exit(1)

    records = []
    with open(SCORES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 5:
                logger.warning(f"Skipping invalid line: {line}")
                continue

            date, group, home, away, score = parts
            match = re.match(r"(\d+)-(\d+)", score)
            if not match:
                logger.warning(f"Invalid score format: {score}")
                continue

            records.append({
                "date":       date,
                "group":      group,
                "home_team":  home,
                "away_team":  away,
                "home_goals": int(match.group(1)),
                "away_goals": int(match.group(2)),
                "stage":      "Group Stage",
                "tournament": "FIFA World Cup 2026",
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    return df


def save_results(df: pd.DataFrame):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    logger.success(f"Saved {len(df)} matches -> {OUTPUT_FILE}")

    print(f"\n{len(df)} matchs enregistres :\n")
    print(f"{'Date':<12} {'Grp':<5} {'Home':<25} {'Score':<6} {'Away'}")
    print("-" * 65)
    for _, row in df.iterrows():
        print(
            f"{str(row['date'])[:10]:<12} "
            f"{row['group']:<5} "
            f"{row['home_team']:<25} "
            f"{int(row['home_goals'])}-{int(row['away_goals'])}    "
            f"{row['away_team']}"
        )


def update_simulate_script(df: pd.DataFrame):
    if not SIMULATE_SCRIPT.exists():
        logger.warning("simulate.py not found")
        return

    content = SIMULATE_SCRIPT.read_text(encoding="utf-8")

    lines = ["PLAYED_MATCHES = ["]
    for _, row in df[df["stage"] == "Group Stage"].iterrows():
        lines.append(
            f'    {{"group": "{row["group"]}", '
            f'"home": "{row["home_team"]}", '
            f'"away": "{row["away_team"]}", '
            f'"hg": {int(row["home_goals"])}, '
            f'"ag": {int(row["away_goals"])}}},  # {str(row["date"])[:10]}'
        )
    lines.append("]")
    new_played = "\n".join(lines)

    new_content = re.sub(
        r"PLAYED_MATCHES = \[.*?\]",
        new_played,
        content,
        flags=re.DOTALL
    )

    SIMULATE_SCRIPT.write_text(new_content, encoding="utf-8")
    logger.success(f"Updated PLAYED_MATCHES in simulate.py")


def run_simulation():
    logger.info("Relaunching Monte Carlo simulation...")
    subprocess.run([sys.executable, str(SIMULATE_SCRIPT)])
    logger.success("Simulation complete! Refresh the dashboard.")


def main():
    logger.info("=== WC2026 Live Update ===")

    df = parse_scores()

    if df.empty:
        logger.warning("No valid scores found in scores.txt")
        return

    save_results(df)
    update_simulate_script(df)
    run_simulation()

    print("\n✅ Done! Refresh the dashboard to see updated predictions.")


if __name__ == "__main__":
    main()