"""
collect_squads.py
-----------------
Scrape les squads officiels WC2026 depuis Wikipedia.
Produit : data/raw/squads/wc2026_squads.csv
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from pathlib import Path
import time
import sys

# ── Config ────────────────────────────────────────────────────────────────────
WIKI_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
OUTPUT_DIR = Path("data/raw/squads")
OUTPUT_FILE = OUTPUT_DIR / "wc2026_squads.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; WC2026Predictor/1.0; "
        "+https://github.com/YOUR_USERNAME/wc2026-predictor)"
    )
}


# ── Logger setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.add("logs/collect_squads.log", rotation="1 MB")


# ── Main ──────────────────────────────────────────────────────────────────────
def fetch_page(url: str) -> BeautifulSoup:
    logger.info(f"Fetching {url}")
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def parse_squads(soup: BeautifulSoup) -> pd.DataFrame:
    """
    Parse toutes les tables de joueurs par équipe.
    Wikipedia structure : h2/h3 = team name, table wikitable = squad
    """
    records = []
    current_team = None

    for tag in soup.find_all(["h2", "h3", "table"]):
        # Détecter le nom d'équipe dans les headers
        if tag.name in ["h2", "h3"]:
            span = tag.find("span", class_="mw-headline")
            if span:
                current_team = span.get_text(strip=True)
                logger.debug(f"Team found: {current_team}")

        # Parser les tables wikitable (= squad lists)
        elif tag.name == "table" and "wikitable" in tag.get("class", []):
            if current_team is None:
                continue

            rows = tag.find_all("tr")
            if not rows:
                continue

            # Header row
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            if not any(h in headers for h in ["Name", "Player", "Pos.", "#"]):
                continue

            for row in rows[1:]:
                cells = row.find_all(["td", "th"])
                if len(cells) < 4:
                    continue

                texts = [c.get_text(strip=True) for c in cells]

                try:
                    record = {
                        "team": current_team,
                        "number": texts[0] if texts else "",
                        "position": texts[1] if len(texts) > 1 else "",
                        "name": texts[2] if len(texts) > 2 else "",
                        "date_of_birth": texts[3] if len(texts) > 3 else "",
                        "caps": texts[4] if len(texts) > 4 else "",
                        "goals": texts[5] if len(texts) > 5 else "",
                        "club": texts[6] if len(texts) > 6 else "",
                    }
                    records.append(record)
                except Exception as e:
                    logger.warning(f"Row parse error for {current_team}: {e}")
                    continue

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} players across {df['team'].nunique()} teams")
    return df


def clean_squads(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage basique du DataFrame."""
    # Supprimer les lignes vides ou sans nom
    df = df[df["name"].str.len() > 1].copy()

    # Nettoyer les positions
    position_map = {
        "GK": "GK", "G": "GK",
        "DF": "DF", "D": "DF",
        "MF": "MF", "M": "MF",
        "FW": "FW", "F": "FW", "A": "FW",
    }
    df["position_clean"] = df["position"].map(position_map).fillna(df["position"])

    # Extraire l'année de naissance
    df["birth_year"] = df["date_of_birth"].str.extract(r"(\d{4})").astype("Int64")
    df["age_at_wc"] = 2026 - df["birth_year"]

    # Convertir caps / goals en numérique
    df["caps"] = pd.to_numeric(df["caps"], errors="coerce").astype("Int64")
    df["goals"] = pd.to_numeric(df["goals"], errors="coerce").astype("Int64")

    # Reset index
    df = df.reset_index(drop=True)

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    soup = fetch_page(WIKI_URL)
    time.sleep(1)  # Be polite

    df_raw = parse_squads(soup)
    df_clean = clean_squads(df_raw)

    df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logger.success(f"Saved {len(df_clean)} players → {OUTPUT_FILE}")

    # Quick summary
    summary = df_clean.groupby("team").size().reset_index(name="n_players")
    logger.info(f"\nTeams with < 20 players (might need review):\n"
                f"{summary[summary['n_players'] < 20].to_string()}")

    print("\n📊 Sample data:")
    print(df_clean.head(10).to_string())
    print(f"\n✅ Total: {len(df_clean)} joueurs, {df_clean['team'].nunique()} équipes")


if __name__ == "__main__":
    main()
