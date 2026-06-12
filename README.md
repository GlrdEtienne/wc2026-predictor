# 🏆 WC2026 Predictor

Projet de data science complet pour prédire le déroulement de la **FIFA World Cup 2026** (USA / Canada / Mexique).

## 🎯 Objectifs

- Collecter les stats individuelles des ~1 248 joueurs participants (48 équipes × 26 joueurs)
- Agréger ces stats au niveau équipe pour construire des features de match
- Entraîner un modèle de prédiction de score (XGBoost)
- Simuler la compétition en Monte Carlo (10 000 itérations)
- Visualiser les prédictions dans un dashboard Streamlit interactif

## 📊 Sources de données

| Source | Données | Accès |
|--------|---------|-------|
| Wikipedia | Squads officiels WC2026 (48 équipes) | Scraping gratuit |
| FBref (via `soccerdata`) | Stats joueurs 2025-26 (xG, passes, duels...) | Gratuit (rate-limited) |
| GitHub martj42 | Résultats internationaux historiques | CSV public |
| API-Football | Live scores WC2026 | Free tier (100 req/day) |

## 🗂️ Structure du projet

```
wc2026-predictor/
│
├── src/
│   ├── collect/                  # Scripts de collecte de données
│   │   ├── collect_squads.py         # Squads officiels WC2026
│   │   ├── collect_player_stats.py   # Stats club 2025-26 (FBref)
│   │   └── collect_rankings_and_history.py  # FIFA ranking + historique
│   │
│   ├── features/                 # Feature engineering
│   │   └── build_features.py         # Agrégation joueur → équipe
│   │
│   ├── model/                    # Modèles ML
│   │   ├── train.py                  # Entraînement XGBoost
│   │   └── simulate.py               # Monte Carlo simulation
│   │
│   └── dashboard/                # Visualisation
│       └── app.py                    # Streamlit app
│
├── data/
│   ├── raw/                      # Données brutes (non versionnées)
│   │   ├── squads/
│   │   ├── player_stats/
│   │   └── historical/
│   └── processed/                # Features engineered (versionnées)
│
├── notebooks/                    # Exploration & analyse
├── tests/                        # Tests unitaires
├── config.yaml                   # Configuration centralisée
├── requirements.txt
├── Makefile                      # Commandes projet
└── .env.example                  # Template variables d'environnement
```

## 🚀 Quick Start

### 1. Cloner et installer

```bash
git clone https://github.com/YOUR_USERNAME/wc2026-predictor.git
cd wc2026-predictor
make setup
```

### 2. Configurer les API keys

```bash
cp .env.example .env
# Éditer .env et ajouter ta RAPIDAPI_KEY
```

### 3. Collecter les données

```bash
# Tout en une fois
make collect-all

# Ou étape par étape
make collect-squads     # Squads WC2026 (Wikipedia)
make collect-history    # FIFA ranking + résultats historiques
make collect-stats      # Stats joueurs FBref (long ~30 min)
```

### 4. Entraîner le modèle

```bash
make features   # Feature engineering
make train      # Entraînement XGBoost
make simulate   # Simulation Monte Carlo
```

### 5. Lancer le dashboard

```bash
make run-dashboard
# → http://localhost:8501
```

## 🤖 Modèle

### Architecture

1. **Score prediction** : XGBoost/LightGBM qui prédit les buts de chaque équipe par match
2. **Monte Carlo simulation** : 10 000 simulations de la compétition complète

### Features principales

**Au niveau équipe :**
- xG moyen pour/contre par 90 min
- Pressing intensity (nb pressures/90)
- Passes progressives moyennes
- Age moyen du squad + expérience (caps moyens)
- Ranking FIFA (ecart entre les deux équipes)

**Au niveau match :**
- H2H sur les 5 derniers matchs
- Forme récente (W/D/L 10 derniers matchs)
- Jours de repos depuis dernier match
- Phase de la compétition

### Données d'entraînement

- WC 2010, 2014, 2018, 2022 (phase de groupes + élimination directe)
- Matchs internationaux A (2018-2026)

## 📈 Dashboard

- **Vue Groupes** : classements en temps réel + prédictions des matchs à venir
- **Bracket éliminatoire** : arbre interactif avec probabilités par équipe
- **Fiche équipe** : stats agrégées + meilleurs joueurs
- **Proba de victoire finale** : top 10 avec graphique

## 🏅 Résultats WC2026 (en cours)

| Date | Groupe | Match | Score |
|------|--------|-------|-------|
| 11/06 | A | 🇲🇽 Mexique vs Afrique du Sud 🇿🇦 | **2-0** |
| 11/06 | A | 🇰🇷 Corée du Sud vs Tchéquie 🇨🇿 | **2-1** |

## 📝 License

MIT
