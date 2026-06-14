"""
collect_fixtures.py — v2
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from pathlib import Path
import re, sys, time

OUTPUT_FILE = Path("data/raw/historical/wc2026_schedule.csv")
WIKI_API    = "https://en.wikipedia.org/w/api.php"
HEADERS     = {"User-Agent": "WC2026Predictor/1.0"}
GROUPS      = list("ABCDEFGHIJKL")

TEAM_NAME_MAP = {
    "Korea Republic":     "South Korea",
    "Czechia":            "Czech Republic",
    "Türkiye":            "Turkey",
    "IR Iran":            "Iran",
    "Côte d'Ivoire":      "Ivory Coast",
    "Congo DR":           "DR Congo",
}

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {message}")


def clean_team(name: str) -> str:
    name = re.sub(r"\(.*?\)", "", name).strip()
    return TEAM_NAME_MAP.get(name, name)


def fetch_group_page(group: str) -> BeautifulSoup:
    params = {
        "action": "parse",
        "page": f"2026_FIFA_World_Cup_Group_{group}",
        "prop": "text", "formatversion": "2", "format": "json"
    }
    resp = requests.get(WIKI_API, headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.json()["parse"]["text"], "lxml")


def parse_footballbox(soup: BeautifulSoup, group: str) -> list:
    records = []

    for fbox in soup.find_all("div", class_="footballbox"):
        try:
            # Date — span.bday contient "2026-06-11"
            date_span = fbox.find("span", class_="bday")
            date_str  = date_span.get_text(strip=True) if date_span else ""

            # Heure — div.ftime
            time_el  = fbox.find("div", class_="ftime")
            time_str = time_el.get_text(strip=True) if time_el else ""

            # Équipes — th.fhome et th.faway, prendre le span itemprop="name"
            home_el   = fbox.find("th", class_="fhome")
            away_el   = fbox.find("th", class_="faway")
            home_span = home_el.find("span", itemprop="name") if home_el else None
            away_span = away_el.find("span", itemprop="name") if away_el else None
            home_team = clean_team(home_span.get_text(strip=True)) if home_span else ""
            away_team = clean_team(away_span.get_text(strip=True)) if away_span else ""

            # Score — th.fscore
            score_el   = fbox.find("th", class_="fscore")
            score_text = score_el.get_text(strip=True) if score_el else ""
            sm = re.search(r"(\d+)\s*[–\-]\s*(\d+)", score_text)
            if sm:
                home_goals = int(sm.group(1))
                away_goals = int(sm.group(2))
                status     = "finished"
            else:
                home_goals = None
                away_goals = None
                status     = "scheduled"

            # Stade — div itemprop="location"
            loc_el  = fbox.find("div", itemprop="location")
            stadium = loc_el.get_text(strip=True) if loc_el else ""

            if home_team and away_team:
                records.append({
                    "date":       date_str,
                    "time_local": time_str,
                    "group":      group,
                    "home_team":  home_team,
                    "away_team":  away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "status":     status,
                    "stadium":    stadium,
                })

        except Exception as e:
            logger.debug(f"Parse error group {group}: {e}")

    return records


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_records = []

    for group in GROUPS:
        logger.info(f"Fetching Group {group}...")
        try:
            soup    = fetch_group_page(group)
            records = parse_footballbox(soup, group)
            logger.info(f"  Group {group}: {len(records)} matches")
            all_records.extend(records)
            time.sleep(3)
        except Exception as e:
            logger.error(f"  Group {group} failed: {e}")
            time.sleep(5)

    if not all_records:
        logger.error("No matches found!")
        return

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    finished  = df[df["status"] == "finished"]
    scheduled = df[df["status"] == "scheduled"]

    logger.info(f"Saved {len(df)} matches")
    print(f"\nTotal    : {len(df)}")
    print(f"Finished : {len(finished)}")
    print(f"Scheduled: {len(scheduled)}")
    print("\nSample:")
    print(df[["date", "group", "home_team", "away_team", "home_goals", "away_goals", "status"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()