"""
collect_player_stats.py
-----------------------
Collecte les stats individuelles saison 2025-26 depuis FBref
pour tous les joueurs présents dans les squads WC2026.

Sources : FBref via la lib `soccerdata`
Produit  : data/raw/player_stats/fbref_stats_2025_26.csv
"""

import pandas as pd
import soccerdata as sd
from loguru import logger
from pathlib import Path
import sys
import yaml
import time

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

SQUADS_FILE = Path("data/raw/squads/wc2026_squads.csv")
OUTPUT_DIR  = Path("data/raw/player_stats")
OUTPUT_FILE = OUTPUT_DIR / "fbref_stats_2025_26.csv"

LEAGUES = CONFIG["data"]["leagues_to_scrape"]
SEASON  = "2526"  # Format soccerdata : "2526" = 2025-26

STAT_TYPES = [
    "standard",       # goals, assists, xG, minutes
    "shooting",       # shots, xG, npxG
    "passing",        # passes, key passes, progressive passes
    "defense",        # tackles, interceptions, pressures
    "possession",     # touches, carries, progressive carries
]

# ── Logger ────────────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.add("logs/collect_player_stats.log", rotation="1 MB")


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_wc_player_names(min_caps: int = 0) -> set[str]:
    """Charge la liste des noms de joueurs WC2026."""
    if not SQUADS_FILE.exists():
        logger.warning("Squads file not found — run collect_squads.py first")
        return set()
    df = pd.read_csv(SQUADS_FILE)
    names = set(df["name"].dropna().str.strip())
    logger.info(f"Loaded {len(names)} player names from WC2026 squads")
    return names


def fetch_league_stats(league: str, season: str, stat_type: str) -> pd.DataFrame | None:
    """Fetch une stat table FBref pour une ligue donnée."""
    try:
        fbref = sd.FBref(leagues=[league], seasons=[season])
        df = fbref.read_player_season_stats(stat_type=stat_type)
        df["league"] = league
        df["stat_type"] = stat_type
        logger.info(f"  ✓ {league} | {stat_type} | {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"  ✗ {league} | {stat_type} | Error: {e}")
        return None


def merge_stat_types(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge toutes les stat tables sur player + team + league."""
    if not dfs:
        return pd.DataFrame()

    # Colonnes clés pour le merge
    key_cols = ["player", "team", "league", "nation", "pos", "age", "born", "90s"]

    base = None
    for df in dfs:
        stat_type = df["stat_type"].iloc[0]
        # Drop colonnes redondantes sauf les keys
        stat_cols = [c for c in df.columns if c not in key_cols + ["stat_type"]]
        # Préfixer les colonnes pour éviter les conflits
        rename = {c: f"{stat_type}_{c}" for c in stat_cols}
        df_renamed = df[key_cols + stat_cols].rename(columns=rename)

        if base is None:
            base = df_renamed
        else:
            base = base.merge(df_renamed, on=key_cols, how="outer", suffixes=("", f"_{stat_type}"))

    return base


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    wc_players = load_wc_player_names()

    all_stats: list[pd.DataFrame] = []

    for league in LEAGUES:
        logger.info(f"\n📊 Fetching {league}...")
        league_dfs = []

        for stat_type in STAT_TYPES:
            df = fetch_league_stats(league, SEASON, stat_type)
            if df is not None:
                league_dfs.append(df)
            time.sleep(2)  # Rate limiting FBref — sois poli !

        if league_dfs:
            merged = merge_stat_types(league_dfs)
            all_stats.append(merged)

    if not all_stats:
        logger.error("No data collected. Exiting.")
        return

    # Concat toutes les ligues
    df_all = pd.concat(all_stats, ignore_index=True)
    logger.info(f"\nTotal rows before dedup: {len(df_all)}")

    # Déduplication : garder la saison avec le plus de minutes par joueur
    if "standard_MP" in df_all.columns:
        df_all = (
            df_all
            .sort_values("standard_MP", ascending=False)
            .drop_duplicates(subset=["player", "team"], keep="first")
        )

    # Filtrer sur les joueurs WC2026 (optionnel — garde tous si wc_players vide)
    if wc_players:
        df_wc = df_all[df_all["player"].isin(wc_players)].copy()
        logger.info(f"WC2026 players found in FBref: {len(df_wc)} / {len(wc_players)}")
    else:
        df_wc = df_all.copy()
        logger.warning("No WC player filter applied — keeping all players")

    df_wc.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logger.success(f"Saved {len(df_wc)} player records → {OUTPUT_FILE}")

    print("\n📊 Aperçu des colonnes disponibles:")
    print([c for c in df_wc.columns[:20]])
    print(f"\n✅ {len(df_wc)} joueurs WC2026 avec stats FBref 2025-26")


if __name__ == "__main__":
    main()
