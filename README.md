# WC2026 Predictor

Petit projet de data science autour de la Coupe du Monde 2026 (USA / Canada / Mexique) : collecte des données joueurs et équipes, modèle de prédiction des scores, simulation Monte Carlo de la compétition complète, et un dashboard pour suivre tout ça en direct.

L'idée de base : récupérer les stats des ~1250 joueurs des 48 équipes qualifiées, les agréger au niveau équipe, et voir ce qu'un XGBoost couplé à 10 000 simulations peut dire sur le déroulement du tournoi — phase de groupes et élimination directe.

## Sources de données

- **Wikipedia** — effectifs officiels des 48 équipes et calendrier complet des matchs (scraping via l'API Wikimedia)
- **FBref** (via `soccerdata`) — stats individuelles saison 2025-26 : xG, passes, duels, etc.
- **martj42/international_results** (GitHub) — historique des résultats internationaux depuis 1872
- **scores.txt** — fichier texte mis à jour à la main au fil des matchs pour réactualiser les prédictions


## Installation

Le projet tourne avec Anaconda (numpy/pandas posent souvent des soucis de compilation avec pip seul sous Windows) :

```bash
git clone https://github.com/GlrdEtienne/wc2026-predictor.git
cd wc2026-predictor
conda create -n wc2026 python=3.12
conda activate wc2026
conda install numpy pandas scikit-learn scipy
pip install requests beautifulsoup4 lxml xgboost lightgbm tqdm streamlit plotly altair python-dotenv loguru pyyaml soccerdata
```

## Faire tourner le pipeline

```bash
# Collecte (dans l'ordre, le scraping FBref est le plus long ~15-30 min)
python src/collect/collect_squads.py
python src/collect/collect_rankings_and_history.py
python src/collect/collect_fixtures.py
python src/collect/collect_player_stats.py

# Features + modèle
python src/features/build_features.py
python src/model/train.py
python src/model/simulate.py

# Dashboard
streamlit run src/dashboard/app.py
```

Pour mettre à jour les prédictions après une journée de matchs : éditer `scores.txt`, puis lancer `python src/collect/update_live.py` (ou cliquer sur le bouton dédié dans le dashboard).

## Le modèle

Deux étapes distinctes :

**1. Prédiction du score** — deux XGBoost (un pour les buts à domicile, un pour l'extérieur), entraînés sur l'historique des matchs internationaux 2018-2026 avec une pondération plus forte sur les matchs de Coupe du Monde. Features : écart de ranking FIFA, forme récente, historique tête-à-tête.

**2. Simulation Monte Carlo** — les buts prédits servent de paramètre (lambda) à des tirages de Poisson. La compétition entière est simulée 10 000 fois : phase de groupes, qualification, puis élimination directe jusqu'à la finale. Les résultats déjà connus sont fixés, le reste est tiré aléatoirement. Les probabilités affichées sont juste la fréquence de chaque issue sur les 10 000 simulations.

Précision du modèle sur les matchs de Coupe du Monde historiques : ~62% sur le résultat (victoire/nul/défaite).

La première version de la simulation tournait en boucle Python classique et prenait ~1h40. Vectorisée avec numpy (toutes les prédictions XGBoost pré-calculées en une matrice, tirages de Poisson en bloc), elle tourne maintenant en moins d'une minute.

## Dashboard

- Probabilités de victoire par équipe (phase de groupes → finale)
- Classements de groupe en direct + calendrier des prochains matchs
- Fiche détaillée par équipe (effectif, stats, comparaison au sein du groupe)
- Bouton de mise à jour qui relance la simulation après chaque match

## Limites connues

Pas de prise en compte des blessures, du contexte psychologique, ou de données physiques avancées (non disponibles publiquement). Le modèle reste probabiliste — l'objectif n'est pas de prédire un résultat certain mais de donner une estimation raisonnable, à interpréter comme telle.
