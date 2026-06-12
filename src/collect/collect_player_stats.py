"""
collect_player_stats.py — v3
Stat types corrects + gestion MultiIndex colonnes FBref
"""
import pandas as pd
import soccerdata as sd
from loguru import logger
from pathlib import Path
import sys, time

OUTPUT_DIR  = Path("data/raw/player_stats")
OUTPUT_FILE = OUTPUT_DIR / "fbref_stats_2025_26.csv"
SEASON      = "2526"

LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# Stat types valides pour soccerdata FBref
STAT_TYPES = ["standard", "shooting", "playing_time", "misc"]

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.add("logs/collect_player_stats.log", rotation="1 MB")


def flatten_multiindex(df: pd.DataFrame, stat_type: str) -> pd.DataFrame:
    """Flatten les colonnes MultiIndex de FBref en colonnes simples."""
    if isinstance(df.columns, pd.MultiIndex):
        # Joindre les niveaux : ("Performance", "Gls") → "Gls", ("Per 90", "xG") → "xG"
        # On garde le dernier niveau non-vide
        new_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                # Prendre le dernier niveau non-vide
                parts = [str(c).strip() for c in col if str(c).strip() and str(c) != "nan"]
                new_cols.append("_".join(parts) if len(parts) > 1 else parts[-1] if parts else str(col))
            else:
                new_cols.append(str(col))
        df.columns = new_cols

    # Reset index si player/team sont dans l'index
    if df.index.names and any(n in ["player", "team", "league"] for n in df.index.names):
        df = df.reset_index()

    # Préfixer les colonnes stats (pas les colonnes d'identité)
    id_cols = {"player", "team", "league", "nation", "pos", "age", "born"}
    rename = {}
    for c in df.columns:
        if c not in id_cols:
            rename[c] = f"{stat_type}_{c}"
    df = df.rename(columns=rename)

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Charger noms joueurs WC2026
    squads_file = Path("data/raw/squads/wc2026_squads.csv")
    wc_players = set()
    if squads_file.exists():
        wc_players = set(pd.read_csv(squads_file)["name"].dropna())
        logger.info(f"Loaded {len(wc_players)} WC2026 player names")

    fbref = sd.FBref(leagues=LEAGUES, seasons=[SEASON])

    all_dfs = []
    for stat_type in STAT_TYPES:
        logger.info(f"Fetching {stat_type}...")
        try:
            df = fbref.read_player_season_stats(stat_type=stat_type)
            df = flatten_multiindex(df, stat_type)
            logger.info(f"  ✓ {stat_type}: {len(df)} rows, {len(df.columns)} cols")
            all_dfs.append(df)
            time.sleep(3)
        except Exception as e:
            logger.warning(f"  ✗ {stat_type}: {e}")

    if not all_dfs:
        logger.error("No data collected!")
        return

    # Merge sur player + team + league
    id_cols = ["player", "team", "league"]
    base = all_dfs[0]
    for df in all_dfs[1:]:
        # Garder seulement id_cols + nouvelles colonnes stats
        merge_cols = id_cols + [c for c in df.columns if c not in base.columns]
        base = base.merge(df[merge_cols], on=id_cols, how="outer")

    logger.info(f"Final dataset: {len(base)} rows, {len(base.columns)} columns")

    base.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logger.success(f"Saved → {OUTPUT_FILE}")

    if wc_players:
        found = base[base["player"].isin(wc_players)]
        logger.info(f"WC2026 players found in Big 5: {len(found)} / {len(wc_players)}")
        logger.info("(Remaining play in other leagues: MLS, Saudi Pro, etc.)")

    print(f"\n✅ {len(base)} joueurs | {len(base.columns)} features")
    print(f"   Colonnes: {list(base.columns[:15])}")

if __name__ == "__main__":
    main()