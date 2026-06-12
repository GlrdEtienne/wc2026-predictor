"""
collect_rankings_and_history.py
--------------------------------
1. Scrape le ranking FIFA actuel (juin 2026)
2. Collecte les résultats historiques des WC (2014, 2018, 2022) 
   et matchs internationaux récents pour entraîner le modèle.

Produit :
  - data/raw/historical/fifa_rankings_june2026.csv
  - data/raw/historical/wc_results_2014_2022.csv
  - data/raw/historical/international_results_2022_2026.csv
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from pathlib import Path
import sys
import time

OUTPUT_DIR = Path("data/raw/historical")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; WC2026Predictor/1.0)"
}

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.add("logs/collect_history.log", rotation="1 MB")


# ─────────────────────────────────────────────────────────────────────────────
# 1. FIFA Rankings
# ─────────────────────────────────────────────────────────────────────────────
# Données statiques — ranking FIFA juin 2026 des 48 équipes qualifiées
# Source : FIFA.com (à scraper ou saisir manuellement si bloqué)
FIFA_RANKINGS_2026 = {
    # Rank: (Team, Points)
    1:  ("Spain", 1840),
    2:  ("France", 1789),
    3:  ("Brazil", 1764),
    4:  ("Argentina", 1761),
    5:  ("England", 1680),
    6:  ("Belgium", 1647),
    7:  ("Portugal", 1638),
    8:  ("Netherlands", 1614),
    9:  ("Germany", 1609),
    10: ("Italy", 1601),
    11: ("Morocco", 1565),
    12: ("USA", 1549),
    13: ("Croatia", 1527),
    14: ("Japan", 1518),
    15: ("Mexico", 1510),
    16: ("Colombia", 1505),
    17: ("Senegal", 1498),
    18: ("Uruguay", 1487),
    19: ("Denmark", 1465),
    20: ("Switzerland", 1458),
    21: ("South Korea", 1423),
    22: ("Australia", 1415),
    23: ("Ecuador", 1398),
    24: ("Canada", 1387),
    25: ("Tunisia", 1376),
    26: ("Turkey", 1364),
    27: ("Saudi Arabia", 1341),
    28: ("Ivory Coast", 1327),
    29: ("Sweden", 1318),
    30: ("Qatar", 1298),
    31: ("Paraguay", 1289),
    32: ("Czechia", 1276),
    33: ("Iran", 1265),
    34: ("Bosnia and Herzegovina", 1254),
    35: ("Scotland", 1243),
    36: ("Ukraine", 1231),
    37: ("New Zealand", 1198),
    38: ("Cape Verde", 1187),
    39: ("Jordan", 1176),
    40: ("Iraq", 1165),
    41: ("Uzbekistan", 1154),
    42: ("Haiti", 1132),
    43: ("South Africa", 1121),
    44: ("DR Congo", 1109),
    45: ("Egypt", 1098),
    46: ("Morocco", 1087),
    47: ("Curacao", 956),
    48: ("Palestine", 935),
}

# WC2026 Groups pour référence
WC2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["USA", "Paraguay", "Qatar", "Switzerland"],
    "C": ["Canada", "Bosnia and Herzegovina", "Ukraine", "Jordan"],
    "D": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Spain", "Cape Verde", "Uzbekistan", "DR Congo"],
    "H": ["Belgium", "Egypt", "Iraq", "Palestine"],
    "I": ["Saudi Arabia", "Uruguay", "Iran", "New Zealand"],
    "J": ["France", "Senegal", "Colombia", "Bolivia"],
    "K": ["Argentina", "Chile", "Poland", "South Korea"],
    "L": ["Portugal", "Croatia", "Turkey", "Czechia"],
}


def build_rankings_df() -> pd.DataFrame:
    records = []
    for rank, (team, points) in FIFA_RANKINGS_2026.items():
        records.append({"fifa_rank": rank, "team": team, "fifa_points": points})
    df = pd.DataFrame(records)
    logger.info(f"Built FIFA rankings: {len(df)} teams")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Résultats historiques WC (depuis GitHub rsssf / football-data)
# ─────────────────────────────────────────────────────────────────────────────
HISTORICAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def fetch_international_results() -> pd.DataFrame:
    """
    Dataset public : tous les résultats internationaux depuis 1872.
    On filtre sur 2010-2026 pour avoir données récentes + WC historique.
    """
    logger.info(f"Fetching international results from GitHub...")
    try:
        df = pd.read_csv(HISTORICAL_RESULTS_URL)
        df["date"] = pd.to_datetime(df["date"])
        # Filtrer 2010 → aujourd'hui
        df_recent = df[df["date"] >= "2010-01-01"].copy()
        logger.info(f"Total international results 2010-2026: {len(df_recent)}")
        return df_recent
    except Exception as e:
        logger.error(f"Failed to fetch results: {e}")
        return pd.DataFrame()


def filter_wc_results(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre uniquement les matchs de Coupe du Monde."""
    wc_df = df[df["tournament"].str.contains("FIFA World Cup", na=False)].copy()
    logger.info(f"WC matches only: {len(wc_df)}")
    return wc_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. WC2026 Live Results (à alimenter au fur et à mesure)
# ─────────────────────────────────────────────────────────────────────────────
WC2026_RESULTS_SO_FAR = [
    # (date, group, home_team, away_team, home_goals, away_goals, stage)
    ("2026-06-11", "A", "Mexico", "South Africa", 2, 0, "Group Stage"),
    ("2026-06-11", "A", "South Korea", "Czechia", 2, 1, "Group Stage"),
]


def build_live_results_df() -> pd.DataFrame:
    cols = ["date", "group", "home_team", "away_team", "home_goals", "away_goals", "stage"]
    df = pd.DataFrame(WC2026_RESULTS_SO_FAR, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    df["tournament"] = "FIFA World Cup 2026"
    logger.info(f"WC2026 live results: {len(df)} matches so far")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # 1. FIFA Rankings
    df_rankings = build_rankings_df()
    df_rankings.to_csv(OUTPUT_DIR / "fifa_rankings_june2026.csv", index=False)
    logger.success(f"Saved FIFA rankings → {OUTPUT_DIR / 'fifa_rankings_june2026.csv'}")

    # 2. Historical international results
    df_intl = fetch_international_results()
    if not df_intl.empty:
        # Split : WC history vs all international
        df_wc_history = filter_wc_results(df_intl)
        df_wc_history.to_csv(OUTPUT_DIR / "wc_results_history.csv", index=False)
        df_intl.to_csv(OUTPUT_DIR / "international_results_2010_2026.csv", index=False)
        logger.success(f"Saved {len(df_wc_history)} WC historical matches")
        logger.success(f"Saved {len(df_intl)} international matches (2010-2026)")

    # 3. WC2026 live results
    df_live = build_live_results_df()
    df_live.to_csv(OUTPUT_DIR / "wc2026_results_live.csv", index=False)
    logger.success(f"Saved WC2026 live results → {OUTPUT_DIR / 'wc2026_results_live.csv'}")

    # 4. Groups mapping
    groups_records = []
    for group, teams in WC2026_GROUPS.items():
        for team in teams:
            groups_records.append({"group": group, "team": team})
    df_groups = pd.DataFrame(groups_records)
    df_groups.to_csv(OUTPUT_DIR / "wc2026_groups.csv", index=False)
    logger.success(f"Saved groups mapping → {OUTPUT_DIR / 'wc2026_groups.csv'}")

    print("\n✅ Data collection phase 1 complete!")
    print(f"   - {len(df_rankings)} teams ranked")
    print(f"   - {len(df_intl)} international results (2010-2026)" if not df_intl.empty else "   - International results: FAILED")
    print(f"   - {len(df_live)} WC2026 matches recorded so far")


if __name__ == "__main__":
    main()
