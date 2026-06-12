"""
simulate.py
-----------
Monte Carlo simulation de la WC2026.
10 000 simulations → probabilités par équipe pour chaque stade.

Produit :
  - data/processed/simulation_results.csv
  - data/processed/group_stage_probs.csv
  - data/processed/knockout_probs.csv
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from loguru import logger
from scipy.stats import poisson
import sys
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb

PROC   = Path("data/processed")
MODELS = Path("models")

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

N_SIMULATIONS = 10_000
RANDOM_SEED   = 42

# WC2026 groupes
WC2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["United States", "Paraguay", "Qatar", "Switzerland"],
    "C": ["Canada", "Bosnia and Herzegovina", "Ukraine", "Jordan"],
    "D": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Spain", "Cape Verde", "Uzbekistan", "DR Congo"],
    "H": ["Belgium", "Egypt", "Iraq", "Norway"],
    "I": ["Saudi Arabia", "Uruguay", "Iran", "New Zealand"],
    "J": ["France", "Senegal", "Colombia", "Algeria"],
    "K": ["Argentina", "Austria", "Panama", "Ghana"],
    "L": ["Portugal", "Croatia", "Turkey", "England"],
}

# Résultats déjà joués (à updater au fur et à mesure)
PLAYED_MATCHES = [
    {"group": "A", "home": "Mexico",      "away": "South Africa", "hg": 2, "ag": 0},
    {"group": "A", "home": "South Korea", "away": "Czech Republic", "hg": 2, "ag": 1},
]


def load_models_and_features():
    home_model = xgb.XGBRegressor()
    home_model.load_model(str(MODELS / "home_goals_model.json"))

    away_model = xgb.XGBRegressor()
    away_model.load_model(str(MODELS / "away_goals_model.json"))

    with open(MODELS / "feature_columns.json") as f:
        features = json.load(f)

    return home_model, away_model, features


def load_team_data():
    team_feat = pd.read_csv(PROC / "team_features.csv")
    rank_map  = team_feat.set_index("team")["fifa_rank"].to_dict()
    pts_map   = team_feat.set_index("team")["fifa_points"].to_dict()
    form_map  = team_feat.set_index("team").get("avg_form", pd.Series(dtype=float)).to_dict()
    return rank_map, pts_map, form_map, team_feat


def predict_goals(home_team, away_team, rank_map, pts_map, form_map,
                  home_model, away_model, features, is_wc=1):
    """Prédit les buts moyens attendus pour un match."""

    hr = rank_map.get(home_team, 50)
    ar = rank_map.get(away_team, 50)
    hp = pts_map.get(home_team, 1200)
    ap = pts_map.get(away_team, 1200)
    hf = form_map.get(home_team, 1.5)
    af = form_map.get(away_team, 1.5)

    row = {
        "home_fifa_rank":   hr,
        "away_fifa_rank":   ar,
        "fifa_rank_diff":   hr - ar,
        "fifa_points_diff": hp - ap,
        "home_form_pts":    hf,
        "away_form_pts":    af,
        "home_h2h_wins":    0,
        "away_h2h_wins":    0,
        "is_wc":            is_wc,
    }

    X = pd.DataFrame([row])[features]
    pred_home = float(home_model.predict(X)[0])
    pred_away = float(away_model.predict(X)[0])

    # Clamp valeurs raisonnables
    pred_home = max(0.2, min(pred_home, 6.0))
    pred_away = max(0.2, min(pred_away, 6.0))

    return pred_home, pred_away


def simulate_match(lambda_home, lambda_away, rng):
    """Simule un match via distribution de Poisson."""
    hg = rng.poisson(lambda_home)
    ag = rng.poisson(lambda_away)
    return hg, ag


def simulate_group(teams, rank_map, pts_map, form_map,
                   home_model, away_model, features,
                   played: list, rng) -> pd.DataFrame:
    """Simule une phase de groupes et retourne le classement."""
    from itertools import combinations

    # Init tableau
    standings = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "w": 0, "d": 0, "l": 0} for t in teams}

    # Résultats déjà joués
    played_set = {(m["home"], m["away"]): (m["hg"], m["ag"]) for m in played}

    for home, away in combinations(teams, 2):
        if (home, away) in played_set:
            hg, ag = played_set[(home, away)]
        else:
            lh, la = predict_goals(home, away, rank_map, pts_map, form_map,
                                   home_model, away_model, features)
            hg, ag = simulate_match(lh, la, rng)

        # Update standings
        standings[home]["gf"] += hg
        standings[home]["ga"] += ag
        standings[away]["gf"] += ag
        standings[away]["ga"] += hg
        standings[home]["gd"] = standings[home]["gf"] - standings[home]["ga"]
        standings[away]["gd"] = standings[away]["gf"] - standings[away]["ga"]

        if hg > ag:
            standings[home]["pts"] += 3
            standings[home]["w"]   += 1
            standings[away]["l"]   += 1
        elif ag > hg:
            standings[away]["pts"] += 3
            standings[away]["w"]   += 1
            standings[home]["l"]   += 1
        else:
            standings[home]["pts"] += 1
            standings[away]["pts"] += 1
            standings[home]["d"]   += 1
            standings[away]["d"]   += 1

    # Classement : pts, gd, gf, puis aléatoire pour égalité parfaite
    df = pd.DataFrame(standings).T.reset_index().rename(columns={"index": "team"})
    df["rand"] = rng.random(len(df))
    df = df.sort_values(["pts", "gd", "gf", "rand"], ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def simulate_knockout_match(home_team, away_team, rank_map, pts_map, form_map,
                             home_model, away_model, features, rng) -> str:
    """Simule un match éliminatoire (avec prolongations/penalties si nul)."""
    lh, la = predict_goals(home_team, away_team, rank_map, pts_map, form_map,
                           home_model, away_model, features)
    hg, ag = simulate_match(lh, la, rng)

    if hg != ag:
        return home_team if hg > ag else away_team

    # Prolongations — légèrement favorise le mieux classé
    hr = rank_map.get(home_team, 50)
    ar = rank_map.get(away_team, 50)
    # Probabilité de gagner les penalties basée sur le ranking
    home_pen_prob = 0.5 + (ar - hr) * 0.003
    home_pen_prob = max(0.3, min(0.7, home_pen_prob))

    return home_team if rng.random() < home_pen_prob else away_team


def run_simulations(home_model, away_model, features, rank_map, pts_map, form_map):
    rng = np.random.default_rng(RANDOM_SEED)

    # Compteurs de résultats
    group_qualif  = {t: 0 for group in WC2026_GROUPS.values() for t in group}
    r32_count     = {t: 0 for group in WC2026_GROUPS.values() for t in group}
    qf_count      = {t: 0 for group in WC2026_GROUPS.values() for t in group}
    sf_count      = {t: 0 for group in WC2026_GROUPS.values() for t in group}
    final_count   = {t: 0 for group in WC2026_GROUPS.values() for t in group}
    winner_count  = {t: 0 for group in WC2026_GROUPS.values() for t in group}

    logger.info(f"Running {N_SIMULATIONS:,} simulations...")

    for sim in range(N_SIMULATIONS):
        if sim % 1000 == 0:
            logger.info(f"  Simulation {sim:,}/{N_SIMULATIONS:,}...")

        # ── Phase de groupes ──────────────────────────────────────────────────
        group_winners  = {}
        group_runners  = {}
        third_places   = []  # (team, pts, gd, gf) pour sélectionner les 8 meilleurs 3èmes

        for group_name, teams in WC2026_GROUPS.items():
            played = [m for m in PLAYED_MATCHES if m["group"] == group_name]
            standings = simulate_group(teams, rank_map, pts_map, form_map,
                                       home_model, away_model, features, played, rng)

            winner = standings.iloc[0]["team"]
            runner = standings.iloc[1]["team"]
            third  = standings.iloc[2]

            group_winners[group_name] = winner
            group_runners[group_name] = runner
            third_places.append({
                "team": third["team"],
                "group": group_name,
                "pts": third["pts"],
                "gd": third["gd"],
                "gf": third["gf"],
            })

            group_qualif[winner] += 1
            group_qualif[runner] += 1

        # 8 meilleurs 3èmes
        third_df = pd.DataFrame(third_places).sort_values(
            ["pts", "gd", "gf"], ascending=False
        ).head(8)
        best_thirds = list(third_df["team"])
        for t in best_thirds:
            group_qualif[t] += 1

        # ── Round of 32 ───────────────────────────────────────────────────────
        # Bracket WC2026 : 1er groupe vs 2ème autre groupe + meilleurs 3èmes
        # Simplifié : on tire au sort les matchups R32
        qualifiers = list(group_winners.values()) + list(group_runners.values()) + best_thirds
        rng.shuffle(qualifiers)

        r32_winners = []
        for i in range(0, len(qualifiers), 2):
            if i + 1 < len(qualifiers):
                w = simulate_knockout_match(
                    qualifiers[i], qualifiers[i+1],
                    rank_map, pts_map, form_map,
                    home_model, away_model, features, rng
                )
                r32_winners.append(w)
                r32_count[qualifiers[i]] += 1
                r32_count[qualifiers[i+1]] += 1

        # ── Quarts de finale ──────────────────────────────────────────────────
        rng.shuffle(r32_winners)
        qf_winners = []
        for i in range(0, len(r32_winners), 2):
            if i + 1 < len(r32_winners):
                w = simulate_knockout_match(
                    r32_winners[i], r32_winners[i+1],
                    rank_map, pts_map, form_map,
                    home_model, away_model, features, rng
                )
                qf_winners.append(w)
                qf_count[r32_winners[i]] += 1
                qf_count[r32_winners[i+1]] += 1

        # ── Demi-finales ──────────────────────────────────────────────────────
        sf_winners = []
        for i in range(0, len(qf_winners), 2):
            if i + 1 < len(qf_winners):
                w = simulate_knockout_match(
                    qf_winners[i], qf_winners[i+1],
                    rank_map, pts_map, form_map,
                    home_model, away_model, features, rng
                )
                sf_winners.append(w)
                sf_count[qf_winners[i]] += 1
                sf_count[qf_winners[i+1]] += 1

        # ── Finale ────────────────────────────────────────────────────────────
        if len(sf_winners) >= 2:
            finalist_a = sf_winners[0]
            finalist_b = sf_winners[1]
            final_count[finalist_a] += 1
            final_count[finalist_b] += 1
            champion = simulate_knockout_match(
                finalist_a, finalist_b,
                rank_map, pts_map, form_map,
                home_model, away_model, features, rng
            )
            winner_count[champion] += 1

    # ── Résultats ─────────────────────────────────────────────────────────────
    all_teams = [t for group in WC2026_GROUPS.values() for t in group]
    results = []
    for team in all_teams:
        group = next(g for g, teams in WC2026_GROUPS.items() if team in teams)
        results.append({
            "team":             team,
            "group":            group,
            "prob_qualify":     round(group_qualif[team]  / N_SIMULATIONS, 4),
            "prob_r32":         round(r32_count[team]     / N_SIMULATIONS, 4),
            "prob_qf":          round(qf_count[team]      / N_SIMULATIONS, 4),
            "prob_sf":          round(sf_count[team]      / N_SIMULATIONS, 4),
            "prob_final":       round(final_count[team]   / N_SIMULATIONS, 4),
            "prob_winner":      round(winner_count[team]  / N_SIMULATIONS, 4),
        })

    return pd.DataFrame(results).sort_values("prob_winner", ascending=False)


def main():
    logger.info("=== Monte Carlo Simulation WC2026 ===")

    home_model, away_model, features = load_models_and_features()
    rank_map, pts_map, form_map, team_feat = load_team_data()

    df_results = run_simulations(home_model, away_model, features, rank_map, pts_map, form_map)

    # Sauvegarder
    df_results.to_csv(PROC / "simulation_results.csv", index=False)
    logger.success(f"Saved simulation_results.csv")

    # Affichage
    print(f"\n🏆 WC2026 WIN PROBABILITY (top 20) — {N_SIMULATIONS:,} simulations\n")
    print(f"{'Team':<25} {'Win%':>6} {'Final%':>7} {'SF%':>6} {'QF%':>6} {'R32%':>6} {'Group%':>7}")
    print("-" * 70)
    for _, row in df_results.head(20).iterrows():
        print(
            f"{row['team']:<25} "
            f"{row['prob_winner']*100:>5.1f}% "
            f"{row['prob_final']*100:>6.1f}% "
            f"{row['prob_sf']*100:>5.1f}% "
            f"{row['prob_qf']*100:>5.1f}% "
            f"{row['prob_r32']*100:>5.1f}% "
            f"{row['prob_qualify']*100:>6.1f}%"
        )


if __name__ == "__main__":
    main()