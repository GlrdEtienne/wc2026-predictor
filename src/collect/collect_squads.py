"""
collect_squads.py — v3
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from pathlib import Path
import re, sys, time

OUTPUT_DIR = Path("data/raw/squads")
OUTPUT_FILE = OUTPUT_DIR / "wc2026_squads.csv"

WIKI_API_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=parse&page=2026_FIFA_World_Cup_squads"
    "&prop=text&formatversion=2&format=json"
)
HEADERS = {"User-Agent": "WC2026Predictor/1.0"}

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.add("logs/collect_squads.log", rotation="1 MB")

SKIP_SECTIONS = {
    "Players", "Outfield players", "Goalkeepers",
    "Player representation by club",
    "Player representation by league system",
    "Player representation by club confederation",
    "Average age of squads",
    "Coach representation by country",
    "Age", "Notes", "References", "See also",
}


def fetch_wiki_html() -> BeautifulSoup:
    logger.info("Fetching Wikipedia via Wikimedia API...")
    resp = requests.get(WIKI_API_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.json()["parse"]["text"]
    logger.info(f"HTML fetched: {len(html)} chars")
    return BeautifulSoup(html, "lxml")


def parse_squads(soup: BeautifulSoup) -> pd.DataFrame:
    records = []

    # Tous les div.mw-heading3 contiennent les noms d'équipes
    heading_divs = soup.find_all("div", class_="mw-heading3")
    logger.info(f"Found {len(heading_divs)} mw-heading3 divs")

    for heading_div in heading_divs:
        h3 = heading_div.find("h3")
        if not h3:
            continue

        team_name = h3.get_text(strip=True)
        if team_name in SKIP_SECTIONS:
            continue

        # La table est un sibling du heading_div
        next_table = None
        for sib in heading_div.find_next_siblings():
            if sib.name == "table" and "wikitable" in sib.get("class", []):
                next_table = sib
                break
            # Stop si on tombe sur le prochain heading
            if sib.name == "div" and any(
                c in sib.get("class", []) for c in ["mw-heading2", "mw-heading3"]
            ):
                break

        if next_table is None:
            logger.debug(f"No table for: {team_name}")
            continue

        rows = next_table.find_all("tr")
        if len(rows) < 2:
            continue

        logger.info(f"  ✓ {team_name}: {len(rows)-1} players")

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 5:
                continue

            texts = [re.sub(r"\[.*?\]", "", c.get_text(strip=True)).strip() for c in cells]

            # Position: "1GK" → "GK"
            pos = re.sub(r"^\d+", "", texts[1]) if len(texts) > 1 else ""

            # Date de naissance depuis span.bday
            dob_span = row.find("span", class_="bday")
            dob = dob_span.get_text(strip=True) if dob_span else ""

            # Nom: enlever "(captain)"
            name = re.sub(r"\(captain\)", "", texts[2]).strip()

            records.append({
                "team":          team_name,
                "number":        texts[0],
                "position":      pos,
                "name":          name,
                "date_of_birth": dob,
                "caps":          texts[4] if len(texts) > 4 else "",
                "goals":         texts[5] if len(texts) > 5 else "",
                "club":          texts[6] if len(texts) > 6 else "",
            })

    df = pd.DataFrame(records)
    logger.info(f"Total: {len(df)} players, {df['team'].nunique() if not df.empty else 0} teams")
    return df


def clean_squads(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["name"].str.len() > 1].copy()

    position_map = {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}
    df["position"] = df["position"].map(position_map).fillna(df["position"])

    df["birth_year"] = pd.to_datetime(df["date_of_birth"], errors="coerce").dt.year.astype("Int64")
    df["age_at_wc"]  = 2026 - df["birth_year"]
    df["caps"]       = pd.to_numeric(df["caps"],  errors="coerce").astype("Int64")
    df["goals"]      = pd.to_numeric(df["goals"], errors="coerce").astype("Int64")

    return df.reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    soup     = fetch_wiki_html()
    df_raw   = parse_squads(soup)

    if df_raw.empty:
        logger.error("No data parsed!")
        return

    df_clean = clean_squads(df_raw)
    df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logger.success(f"Saved {len(df_clean)} players → {OUTPUT_FILE}")

    summary = df_clean.groupby("team").size().reset_index(name="n_players")
    print(f"\n✅ {len(df_clean)} joueurs | {df_clean['team'].nunique()} équipes")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()