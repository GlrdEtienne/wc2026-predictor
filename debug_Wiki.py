"""
debug_wiki_full.py — debug complet en une passe
python debug_wiki_full.py
"""
import requests
from bs4 import BeautifulSoup

WIKI_API_URL = (
    "https://en.wikipedia.org/w/api.php"
    "?action=parse&page=2026_FIFA_World_Cup_squads"
    "&prop=text&formatversion=2&format=json"
)
HEADERS = {"User-Agent": "WC2026Predictor/1.0"}

resp = requests.get(WIKI_API_URL, headers=HEADERS, timeout=30)
html = resp.json()["parse"]["text"]
soup = BeautifulSoup(html, "lxml")

# 1. Trouver le div mw-heading3 Czech Republic
h3 = next(t for t in soup.find_all("h3") if "Czech" in t.get_text())
heading_div = h3.parent

print("=" * 60)
print("1. SIBLINGS DU DIV mw-heading3 (20 premiers)")
print("=" * 60)
for i, sib in enumerate(heading_div.find_next_siblings()):
    print(f"[{i}] {sib.name} | classes={sib.get('class')} | text={sib.get_text(strip=True)[:80]}")
    if i >= 19:
        break

print()
print("=" * 60)
print("2. STRUCTURE COMPLETE AUTOUR DE LA PREMIERE WIKITABLE")
print("=" * 60)
first_table = soup.find("table", class_="wikitable")
print("Table parent tag:", first_table.parent.name)
print("Table parent classes:", first_table.parent.get("class"))
print("Table prev siblings (3):")
for i, sib in enumerate(first_table.find_previous_siblings()):
    print(f"  [{i}] {sib.name} | classes={sib.get('class')} | text={sib.get_text(strip=True)[:80]}")
    if i >= 2:
        break

print()
print("=" * 60)
print("3. TOUTE LA PAGE — tags uniques avec classes")
print("=" * 60)
seen = set()
for tag in soup.find_all(True):
    key = (tag.name, str(tag.get("class")))
    if key not in seen:
        seen.add(key)
        print(f"{tag.name:15} | {str(tag.get('class')):50} | {tag.get_text(strip=True)[:50]}")