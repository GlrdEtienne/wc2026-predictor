"""
simulate.py — v3
Phase éliminatoire vectorisée : tous les lambdas pré-calculés.
Objectif : < 30 secondes.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from loguru import logger
import sys, time
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from itertools import combinations, permutations

PROC   = Path("data/processed")
MODELS = Path("models")

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {message}")

N_SIMULATIONS = 10_000
RANDOM_SEED   = 42

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

PLAYED_MATCHES = [
    {"group": "A", "home": "Mexico", "away": "South Africa", "hg": 2, "ag": 0},  # 2026-06-11
    {"group": "A", "home": "South Korea", "away": "Czech Republic", "hg": 2, "ag": 1},  # 2026-06-11
    {"group": "B", "home": "Canada", "away": "Bosnia and Herzegovina", "hg": 1, "ag": 1},  # 2026-06-12
    {"group": "D", "home": "USA", "away": "Paraguay", "hg": 4, "ag": 1},  # 2026-06-13
    {"group": "B", "home": "Qatar", "away": "Switzerland", "hg": 1, "ag": 1},  # 2026-06-13
    {"group": "E", "home": "Germany", "away": "Curaçao", "hg": 7, "ag": 1},  # 2026-06-14
    {"group": "D", "home": "Australia", "away": "Turkey", "hg": 2, "ag": 0},  # 2026-06-14
    {"group": "F", "home": "Netherlands", "away": "Japan", "hg": 2, "ag": 2},  # 2026-06-14
    {"group": "C", "home": "Brasil", "away": "Morocco", "hg": 1, "ag": 1},  # 2026-06-14
    {"group": "C", "home": "Haiti", "away": "Scotland", "hg": 0, "ag": 1},  # 2026-06-14
    {"group": "E", "home": "Ivory Coast", "away": "Ecuador", "hg": 1, "ag": 0},  # 2026-06-15
    {"group": "F", "home": "Sweden", "away": "Tunisia", "hg": 5, "ag": 1},  # 2026-06-15
    {"group": "H", "home": "Spain", "away": "Cape Verde", "hg": 0, "ag": 0},  # 2026-06-15
    {"group": "H", "home": "Belgium", "away": "Egypt", "hg": 1, "ag": 1},  # 2026-06-15
    {"group": "H", "home": "Saudi Arabia", "away": "Uruguay", "hg": 1, "ag": 1},  # 2026-06-16
    {"group": "G", "home": "Iran", "away": "New Zealand", "hg": 2, "ag": 2},  # 2026-06-16
    {"group": "I", "home": "France", "away": "Senegal", "hg": 3, "ag": 1},  # 2026-06-16
    {"group": "L", "home": "England", "away": "Croatia", "hg": 4, "ag": 2},  # 2026-06-17
    {"group": "K", "home": "Portugal", "away": "DR Congo", "hg": 1, "ag": 1},  # 2026-06-17
    {"group": "J", "home": "Austria", "away": "Jordan", "hg": 3, "ag": 1},  # 2026-06-17
    {"group": "J", "home": "Argentina", "away": "Algeria", "hg": 3, "ag": 0},  # 2026-06-17
    {"group": "I", "home": "Irak", "away": "Norway", "hg": 1, "ag": 4},  # 2026-06-17
    {"group": "L", "home": "Ghana", "away": "Panama", "hg": 1, "ag": 0},  # 2026-06-18
    {"group": "K", "home": "Uzbekistan", "away": "Colombia", "hg": 1, "ag": 3},  # 2026-06-18
    {"group": "A", "home": "Czech Republic", "away": "South Africa", "hg": 1, "ag": 1},  # 2026-06-18
    {"group": "B", "home": "Switzerland", "away": "Bosnia and Herzegovina", "hg": 4, "ag": 1},  # 2026-06-18
    {"group": "B", "home": "Canada", "away": "Qatar", "hg": 6, "ag": 0},  # 2026-06-19
    {"group": "A", "home": "Mexico", "away": "South Korea", "hg": 1, "ag": 0},  # 2026-06-19
    {"group": "D", "home": "USA", "away": "Australia", "hg": 2, "ag": 0},  # 2026-06-19
    {"group": "E", "home": "Germany", "away": "Ivory Coast", "hg": 2, "ag": 1},  # 2026-06-20
    {"group": "F", "home": "Netherlands", "away": "Sweden", "hg": 5, "ag": 1},  # 2026-06-20
    {"group": "C", "home": "Scotland", "away": "Morocco", "hg": 0, "ag": 1},  # 2026-06-20
    {"group": "C", "home": "Brasil", "away": "Haiti", "hg": 3, "ag": 0},  # 2026-06-20
    {"group": "D", "home": "Turkey", "away": "Paraguay", "hg": 0, "ag": 1},  # 2026-06-20
    {"group": "E", "home": "Ecuador", "away": "Curaçao", "hg": 0, "ag": 0},  # 2026-06-21
    {"group": "F", "home": "Tunisia", "away": "Japan", "hg": 0, "ag": 4},  # 2026-06-21
    {"group": "H", "home": "Spain", "away": "Saudi Arabia", "hg": 4, "ag": 0},  # 2026-06-21
    {"group": "G", "home": "Belgium", "away": "Iran", "hg": 0, "ag": 0},  # 2026-06-21
    {"group": "H", "home": "Uruguay", "away": "Cape Verde", "hg": 2, "ag": 2},  # 2026-06-22
    {"group": "G", "home": "New Zealand", "away": "Egypt", "hg": 1, "ag": 3},  # 2026-06-22
]


def load_models_and_features():
    hm = xgb.XGBRegressor(); hm.load_model(str(MODELS / "home_goals_model.json"))
    am = xgb.XGBRegressor(); am.load_model(str(MODELS / "away_goals_model.json"))
    with open(MODELS / "feature_columns.json") as f:
        features = json.load(f)
    return hm, am, features


def load_team_data():
    tf = pd.read_csv(PROC / "team_features.csv")
    rank_map = tf.set_index("team")["fifa_rank"].to_dict()
    pts_map  = tf.set_index("team")["fifa_points"].to_dict()
    return rank_map, pts_map


def predict_lambdas_batch(matchups, rank_map, pts_map, hm, am, features):
    """Prédit les lambdas pour une liste de matchups en une passe."""
    rows = []
    for home, away in matchups:
        hr = rank_map.get(home, 50); ar = rank_map.get(away, 50)
        hp = pts_map.get(home, 1200); ap = pts_map.get(away, 1200)
        rows.append({
            "home_fifa_rank": hr, "away_fifa_rank": ar,
            "fifa_rank_diff": hr - ar, "fifa_points_diff": hp - ap,
            "home_form_pts": 1.5, "away_form_pts": 1.5,
            "home_h2h_wins": 0, "away_h2h_wins": 0, "is_wc": 1,
        })
    X = pd.DataFrame(rows)[features]
    lh = np.clip(hm.predict(X).astype(float), 0.2, 6.0)
    la = np.clip(am.predict(X).astype(float), 0.2, 6.0)
    return lh, la


def simulate_group_stage(all_teams, rank_map, pts_map, hm, am, features, rng):
    """Simule les groupes, retourne (N, 32) qualifiés + (N, 12) 3èmes."""
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    # Pré-calculer tous les matchups de groupes
    group_matchups = [(h, a) for g in WC2026_GROUPS.values() for h, a in combinations(g, 2)]
    lh_grp, la_grp = predict_lambdas_batch(group_matchups, rank_map, pts_map, hm, am, features)
    matchup_to_idx = {(h, a): i for i, (h, a) in enumerate(group_matchups)}
    played_dict    = {(m["home"], m["away"]): (m["hg"], m["ag"]) for m in PLAYED_MATCHES}

    group_winners  = np.zeros((N_SIMULATIONS, 12), dtype=np.int32)
    group_runners  = np.zeros((N_SIMULATIONS, 12), dtype=np.int32)
    thirds_global  = np.zeros((N_SIMULATIONS, 12), dtype=np.int32)
    thirds_pts     = np.zeros((N_SIMULATIONS, 12), dtype=np.int32)
    thirds_gd      = np.zeros((N_SIMULATIONS, 12), dtype=np.int32)
    thirds_gf      = np.zeros((N_SIMULATIONS, 12), dtype=np.int32)

    for gi, (group, teams) in enumerate(WC2026_GROUPS.items()):
        n_t = len(teams)
        t_idx = {t: i for i, t in enumerate(teams)}
        standings = np.zeros((N_SIMULATIONS, n_t, 3), dtype=np.int32)

        for home, away in combinations(teams, 2):
            hi, ai = t_idx[home], t_idx[away]
            if (home, away) in played_dict:
                hg_f, ag_f = played_dict[(home, away)]
                hg = np.full(N_SIMULATIONS, hg_f, dtype=np.int32)
                ag = np.full(N_SIMULATIONS, ag_f, dtype=np.int32)
            else:
                idx = matchup_to_idx[(home, away)]
                hg = rng.poisson(lh_grp[idx], size=N_SIMULATIONS).astype(np.int32)
                ag = rng.poisson(la_grp[idx], size=N_SIMULATIONS).astype(np.int32)

            hw = hg > ag; aw = ag > hg; dr = hg == ag
            standings[:, hi, 0] += np.where(hw, 3, np.where(dr, 1, 0))
            standings[:, ai, 0] += np.where(aw, 3, np.where(dr, 1, 0))
            standings[:, hi, 1] += (hg - ag); standings[:, ai, 1] += (ag - hg)
            standings[:, hi, 2] += hg;        standings[:, ai, 2] += ag

        rand_tb = rng.random((N_SIMULATIONS, n_t))
        score   = standings[:,:,0]*1e6 + standings[:,:,1]*1e3 + standings[:,:,2] + rand_tb*0.1
        ranked  = np.argsort(-score, axis=1)

        group_winners[:, gi] = [team_to_idx[teams[ranked[s, 0]]] for s in range(N_SIMULATIONS)]
        group_runners[:, gi] = [team_to_idx[teams[ranked[s, 1]]] for s in range(N_SIMULATIONS)]
        t3_local             = ranked[:, 2]
        thirds_global[:, gi] = [team_to_idx[teams[t3_local[s]]] for s in range(N_SIMULATIONS)]
        thirds_pts[:, gi]    = standings[np.arange(N_SIMULATIONS), t3_local, 0]
        thirds_gd[:, gi]     = standings[np.arange(N_SIMULATIONS), t3_local, 1]
        thirds_gf[:, gi]     = standings[np.arange(N_SIMULATIONS), t3_local, 2]

    # Top 8 troisièmes
    rand_tb3  = rng.random((N_SIMULATIONS, 12))
    t3_score  = thirds_pts*1e6 + thirds_gd*1e3 + thirds_gf + rand_tb3*0.1
    best8_loc = np.argsort(-t3_score, axis=1)[:, :8]
    best8_gl  = thirds_global[np.arange(N_SIMULATIONS)[:, None], best8_loc]

    # 32 qualifiés : 12 winners + 12 runners + 8 best thirds
    qualified = np.concatenate([group_winners, group_runners, best8_gl], axis=1)
    return qualified


def simulate_knockout_stage(qualified, all_teams, rank_map, pts_map, hm, am, features, rng):
    """
    Phase éliminatoire entièrement vectorisée.
    Clé : pré-calculer les lambdas pour TOUTES les paires possibles (48*47=2256).
    """
    n_teams = len(all_teams)
    team_to_idx = {t: i for i, t in enumerate(all_teams)}

    logger.info("Pre-computing all knockout lambdas (48x47 pairs)...")

    # Tous les matchups ordonnés possibles entre 48 équipes
    all_ko_matchups = [(h, a) for h in all_teams for a in all_teams if h != a]
    lh_ko, la_ko = predict_lambdas_batch(all_ko_matchups, rank_map, pts_map, hm, am, features)

    # Index rapide : (i, j) -> lambda
    lh_matrix = np.zeros((n_teams, n_teams))
    la_matrix = np.zeros((n_teams, n_teams))
    for k, (h, a) in enumerate(all_ko_matchups):
        hi, ai = team_to_idx[h], team_to_idx[a]
        lh_matrix[hi, ai] = lh_ko[k]
        la_matrix[hi, ai] = la_ko[k]

    logger.info(f"Knockout lambdas ready ({len(all_ko_matchups)} pairs)")

    rank_arr = np.array([rank_map.get(t, 50) for t in all_teams])

    r32_count   = np.zeros(n_teams, dtype=np.int32)
    qf_count    = np.zeros(n_teams, dtype=np.int32)
    sf_count    = np.zeros(n_teams, dtype=np.int32)
    final_count = np.zeros(n_teams, dtype=np.int32)
    win_count   = np.zeros(n_teams, dtype=np.int32)

    current = qualified.copy()  # (N, 32)

    for round_name, n_in, counter in [
        ("R32",   32, r32_count),
        ("QF",    16, qf_count),
        ("SF",     8, sf_count),
        ("Final",  4, final_count),
    ]:
        n_matches = n_in // 2
        shuffled  = rng.permuted(current[:, :n_in], axis=1)
        next_round = np.zeros((N_SIMULATIONS, n_matches), dtype=np.int32)

        for m in range(n_matches):
            ta = shuffled[:, m*2]    # (N,)
            tb = shuffled[:, m*2+1]  # (N,)

            # Compter participants
            np.add.at(counter, ta, 1)
            np.add.at(counter, tb, 1)

            # Lambdas vectorisés via matrix lookup
            lh_vec = lh_matrix[ta, tb]  # (N,)
            la_vec = la_matrix[ta, tb]  # (N,)

            # Simuler goals — vectorisé
            # Poisson vectorisé : générer pour chaque sim
            hg = np.array([rng.poisson(l) for l in lh_vec], dtype=np.int32)
            ag = np.array([rng.poisson(l) for l in la_vec], dtype=np.int32)

            winners = np.where(hg > ag, ta, np.where(ag > hg, tb, -1))

            # Penalties pour les nuls
            draw_mask = winners == -1
            if draw_mask.any():
                ra = rank_arr[ta[draw_mask]]
                rb = rank_arr[tb[draw_mask]]
                prob_a = np.clip(0.5 + (rb - ra) * 0.003, 0.3, 0.7)
                pen_rand = rng.random(draw_mask.sum())
                pen_winners = np.where(pen_rand < prob_a, ta[draw_mask], tb[draw_mask])
                winners[draw_mask] = pen_winners

            next_round[:, m] = winners

        current = next_round
        logger.info(f"  {round_name} done")

    win_count += np.bincount(current[:, 0], minlength=n_teams)
    return r32_count, qf_count, sf_count, final_count, win_count


def run_simulations(hm, am, features, rank_map, pts_map):
    rng = np.random.default_rng(RANDOM_SEED)
    all_teams = [t for g in WC2026_GROUPS.values() for t in g]
    team_to_idx = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)

    qualify_count = np.zeros(n_teams, dtype=np.int32)

    logger.info("Simulating group stage...")
    qualified = simulate_group_stage(all_teams, rank_map, pts_map, hm, am, features, rng)

    for i in range(32):
        qualify_count += np.bincount(qualified[:, i], minlength=n_teams)

    logger.info("Simulating knockout stage...")
    r32_c, qf_c, sf_c, fin_c, win_c = simulate_knockout_stage(
        qualified, all_teams, rank_map, pts_map, hm, am, features, rng
    )

    records = []
    for team in all_teams:
        i = team_to_idx[team]
        group = next(g for g, ts in WC2026_GROUPS.items() if team in ts)
        records.append({
            "team":         team, "group": group,
            "prob_qualify": round(qualify_count[i] / N_SIMULATIONS, 4),
            "prob_r32":     round(r32_c[i]         / N_SIMULATIONS, 4),
            "prob_qf":      round(qf_c[i]           / N_SIMULATIONS, 4),
            "prob_sf":      round(sf_c[i]           / N_SIMULATIONS, 4),
            "prob_final":   round(fin_c[i]           / N_SIMULATIONS, 4),
            "prob_winner":  round(win_c[i]           / N_SIMULATIONS, 4),
        })

    return pd.DataFrame(records).sort_values("prob_winner", ascending=False)


def main():
    logger.info("=== Monte Carlo Simulation WC2026 (v3 vectorized) ===")
    t0 = time.time()

    hm, am, features = load_models_and_features()
    rank_map, pts_map = load_team_data()

    df = run_simulations(hm, am, features, rank_map, pts_map)
    df.to_csv(PROC / "simulation_results.csv", index=False)

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.success(f"Saved simulation_results.csv")

    print(f"\nWC2026 WIN PROBABILITY — {N_SIMULATIONS:,} simulations ({elapsed:.1f}s)\n")
    print(f"{'Team':<25} {'Win%':>6} {'Final%':>7} {'SF%':>6} {'QF%':>6}")
    print("-" * 60)
    for _, row in df.head(15).iterrows():
        print(
            f"{row['team']:<25} "
            f"{row['prob_winner']*100:>5.1f}% "
            f"{row['prob_final']*100:>6.1f}% "
            f"{row['prob_sf']*100:>5.1f}% "
            f"{row['prob_qf']*100:>5.1f}%"
        )


if __name__ == "__main__":
    main()