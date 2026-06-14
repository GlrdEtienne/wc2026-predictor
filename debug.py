import requests
from bs4 import BeautifulSoup
WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "WC2026Predictor/1.0"}
params = {"action":"parse","page":"2026_FIFA_World_Cup_Group_A","prop":"text","formatversion":"2","format":"json"}
soup = BeautifulSoup(requests.get(WIKI_API, headers=HEADERS, params=params).json()["parse"]["text"], "lxml")
fbox = soup.find("div", class_="footballbox")
print(fbox)