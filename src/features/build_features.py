"""
build_features.py
-----------------
Agrège les stats individuelles FBref → features au niveau équipe/match.
Produit :
  - data/processed/team_features.csv      (features par équipe WC2026)
  - data/processed/match_features.csv     (features par match historique)
  - data/processed/wc2026_fixtures.csv    (fixtures WC2026 avec features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys, yaml

# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

RAW      = Path("data/raw")
PROC     = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# ── Mapping noms équipes WC2026 → noms dans FBref/historique ─────────────────
# FBref utilise parfois des noms différents
TEAM_NAME_MAP = {
    "United States":        "United States",
    "Czech Republic":       "Czech Republic",
    "Ivory Coast":          "Ivory Coast",
    "DR Congo":             "DR Congo",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "South Korea":          "South Korea",
    "Saudi Arabia":         "Saudi Arabia",
    "New Zealand":          "New Zealand",
    "Cape Verde":           "Cape Verde",
}

# Groupes WC2026
WC2026_GROUPS = CONFIG["tournament"]["groups"]

# Toutes les équipes WC2026
ALL_WC_TEAMS = [team for teams in WC2026_GROUPS.values() for team in teams]

# ── 1. Charger les données ────────────────────────────────────────────────────
def load_data():
    squads      = pd.read_csv(RAW / "squads/wc2026_squads.csv")
    player_stats = pd.read_csv(RAW / "player_stats/fbref_stats_2025_26.csv")
    rankings    = pd.read_csv(RAW / "historical/fifa_rankings_june2026.csv")
    intl        = pd.read_csv(RAW / "historical/international_results_2010_2026.csv", parse_dates=["date"])
    groups      = pd.read_csv(RAW / "historical/wc2026_groups.csv")
    live        = pd.read_csv(RAW / "historical/wc2026_results_live.csv", parse_dates=["date"])

    logger.info(f"Loaded: {len(squads)} squad players, {len(player_stats)} FBref players")
    logger.info(f"Loaded: {len(intl)} international results, {len(rankings)} FIFA rankings")
    return squads, player_stats, rankings, intl, groups, live


# ── 2. Features au niveau équipe ──────────────────────────────────────────────
def build_team_features(squads: pd.DataFrame, player_stats: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Agrège les stats joueurs → features équipe WC2026."""

    # Merge squads avec stats FBref sur le nom du joueur
    merged = squads.merge(
        player_stats,
        left_on="name",
        right_on="player",
        how="left"
    )

    # Colonnes numériques disponibles
    stat_cols = {
        "goals_per90":      "standard_Per 90 Minutes_Gls",
        "assists_per90":    "standard_Per 90 Minutes_Ast",
        "goals_scored":     "standard_Performance_Gls",
        "assists":          "standard_Performance_Ast",
        "minutes_played":   "standard_Playing Time_Min",
        "matches_played":   "standard_Playing Time_MP",
        "yellow_cards":     "standard_Performance_CrdY",
        "red_cards":        "standard_Performance_CrdR",
        "shots_per90":      "shooting_Standard_Sh",
        "shots_on_target":  "shooting_Standard_SoT",
        "xg":               "shooting_Expected_xG",
        "npxg":             "shooting_Expected_npxG",
    }

    # Garder seulement les colonnes qui existent
    available = {k: v for k, v in stat_cols.items() if v in merged.columns}
    logger.info(f"Available stat columns: {list(available.keys())}")

    records = []
    for team in squads["team"].unique():
        team_players = merged[merged["team_x"] == team]
        n_players = len(team_players)

        # Joueurs avec stats FBref (jouent en Big 5)
        with_stats = team_players.dropna(subset=[list(available.values())[0]] if available else [])
        coverage = len(with_stats) / n_players if n_players > 0 else 0

        rec = {
            "team":             team,
            "n_players":        n_players,
            "fbref_coverage":   round(coverage, 2),
            "squad_avg_age":    team_players["age_at_wc"].mean(),
            "squad_avg_caps":   team_players["caps"].mean(),
            "squad_total_goals_intl": team_players["goals"].sum(),
        }

        # Agréger chaque stat (moyenne pondérée par minutes jouées)
        for feat_name, col in available.items():
            vals = pd.to_numeric(team_players[col], errors="coerce").dropna()
            if len(vals) > 0:
                rec[f"avg_{feat_name}"] = vals.mean()
                rec[f"sum_{feat_name}"] = vals.sum()
                rec[f"top3_{feat_name}"] = vals.nlargest(3).mean()  # top 3 joueurs
            else:
                rec[f"avg_{feat_name}"] = np.nan
                rec[f"sum_{feat_name}"] = np.nan
                rec[f"top3_{feat_name}"] = np.nan

        records.append(rec)

    df_team = pd.DataFrame(records)

    # Merge avec FIFA rankings
    df_team = df_team.merge(rankings[["team", "fifa_rank", "fifa_points"]], on="team", how="left")

    logger.info(f"Team features built: {len(df_team)} teams, {len(df_team.columns)} features")
    return df_team


# ── 3. Features au niveau match (historique) ─────────────────────────────────
def build_match_features(intl: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features pour chaque match historique.
    Ces données serviront à entraîner le modèle.
    """
    df = intl.copy()

    # Filtrer sur matchs avec résultats complets
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["home_score", "away_score"])

    # Merge rankings pour les deux équipes
    rankings_dedup = rankings.drop_duplicates(subset=["team"], keep="first")
    rank_map = rankings_dedup.set_index("team")[["fifa_rank", "fifa_points"]].to_dict("index")

    def get_rank(team):
        return rank_map.get(team, {}).get("fifa_rank", 100)

    def get_points(team):
        return rank_map.get(team, {}).get("fifa_points", 1000)

    df["home_fifa_rank"]   = df["home_team"].map(get_rank)
    df["away_fifa_rank"]   = df["away_team"].map(get_rank)
    df["home_fifa_points"] = df["home_team"].map(get_points)
    df["away_fifa_points"] = df["away_team"].map(get_points)
    df["fifa_rank_diff"]   = df["home_fifa_rank"] - df["away_fifa_rank"]
    df["fifa_points_diff"] = df["home_fifa_points"] - df["away_fifa_points"]

    # Features de forme — W/D/L sur les 10 derniers matchs
    logger.info("Computing form features (last 10 matches)...")
    df = df.sort_values("date").reset_index(drop=True)

    form_cache = {}  # team → liste de résultats récents

    home_form, away_form = [], []
    home_wins_h2h, away_wins_h2h = [], []

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        date   = row["date"]

        # Forme home (points sur 10 derniers matchs)
        h_form = form_cache.get(ht, [])
        home_form.append(sum(h_form[-10:]) / max(len(h_form[-10:]), 1))

        a_form = form_cache.get(at, [])
        away_form.append(sum(a_form[-10:]) / max(len(a_form[-10:]), 1))

        # H2H
        h2h = df[
            ((df["home_team"] == ht) & (df["away_team"] == at) |
             (df["home_team"] == at) & (df["away_team"] == ht)) &
            (df["date"] < date)
        ].tail(5)
        h_wins = ((h2h["home_team"] == ht) & (h2h["home_score"] > h2h["away_score"])).sum() + \
                 ((h2h["away_team"] == ht) & (h2h["away_score"] > h2h["home_score"])).sum()
        a_wins = ((h2h["home_team"] == at) & (h2h["home_score"] > h2h["away_score"])).sum() + \
                 ((h2h["away_team"] == at) & (h2h["away_score"] > h2h["home_score"])).sum()
        home_wins_h2h.append(int(h_wins))
        away_wins_h2h.append(int(a_wins))

        # Update form cache
        hs, as_ = row["home_score"], row["away_score"]
        if hs > as_:
            form_cache.setdefault(ht, []).append(3)
            form_cache.setdefault(at, []).append(0)
        elif hs < as_:
            form_cache.setdefault(ht, []).append(0)
            form_cache.setdefault(at, []).append(3)
        else:
            form_cache.setdefault(ht, []).append(1)
            form_cache.setdefault(at, []).append(1)

    df["home_form_pts"]  = home_form
    df["away_form_pts"]  = away_form
    df["home_h2h_wins"]  = home_wins_h2h
    df["away_h2h_wins"]  = away_wins_h2h

    # Target variables
    df["home_goals"] = df["home_score"].astype(int)
    df["away_goals"] = df["away_score"].astype(int)
    df["result"]     = np.where(df["home_goals"] > df["away_goals"], 1,
                       np.where(df["home_goals"] < df["away_goals"], -1, 0))

    # Is WC match
    df["is_wc"] = df["tournament"].str.contains("FIFA World Cup", na=False).astype(int)

    logger.info(f"Match features built: {len(df)} matches, {len(df.columns)} features")
    return df


# ── 4. Fixtures WC2026 avec features ─────────────────────────────────────────
def build_wc2026_fixtures(groups_df: pd.DataFrame, team_features: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """Génère tous les matchs de phase de groupes WC2026 avec leurs features."""
    fixtures = []

    for group, teams in WC2026_GROUPS.items():
        # Tous les matchs possibles dans le groupe (combinaisons 2 à 2)
        from itertools import combinations
        for home, away in combinations(teams, 2):
            fixtures.append({
                "group":     group,
                "stage":     "Group Stage",
                "home_team": home,
                "away_team": away,
            })

    df_fix = pd.DataFrame(fixtures)

    # Merge features équipes
    tf = team_features.set_index("team")

    def get_feat(team, col):
        return tf.loc[team, col] if team in tf.index and col in tf.columns else np.nan

    feat_cols = [c for c in team_features.columns if c not in ["team", "n_players"]]

    for col in feat_cols:
        df_fix[f"home_{col}"] = df_fix["home_team"].map(lambda t: get_feat(t, col))
        df_fix[f"away_{col}"] = df_fix["away_team"].map(lambda t: get_feat(t, col))

    # Diff features
    rank_map = rankings.set_index("team")["fifa_rank"].to_dict()
    df_fix["fifa_rank_diff"]   = df_fix["home_team"].map(rank_map) - df_fix["away_team"].map(rank_map)
    df_fix["home_fifa_rank"]   = df_fix["home_team"].map(rank_map)
    df_fix["away_fifa_rank"]   = df_fix["away_team"].map(rank_map)

    logger.info(f"WC2026 fixtures: {len(df_fix)} group stage matches")
    return df_fix


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logger.info("=== Building Features ===")

    squads, player_stats, rankings, intl, groups, live = load_data()

    # 1. Team features
    logger.info("\n1. Building team features...")
    team_feat = build_team_features(squads, player_stats, rankings)
    team_feat.to_csv(PROC / "team_features.csv", index=False)
    logger.success(f"Saved team_features.csv — {len(team_feat)} teams, {len(team_feat.columns)} features")

    # 2. Match features (historique pour training)
    logger.info("\n2. Building match features (historical)...")
    match_feat = build_match_features(intl, rankings)
    match_feat.to_csv(PROC / "match_features.csv", index=False)
    logger.success(f"Saved match_features.csv — {len(match_feat)} matches")

    # 3. WC2026 fixtures
    logger.info("\n3. Building WC2026 fixtures...")
    fixtures = build_wc2026_fixtures(groups, team_feat, rankings)
    fixtures.to_csv(PROC / "wc2026_fixtures.csv", index=False)
    logger.success(f"Saved wc2026_fixtures.csv — {len(fixtures)} matches")

    print("\n✅ Feature engineering complete!")
    print(f"   team_features.csv  : {len(team_feat)} équipes × {len(team_feat.columns)} features")
    print(f"   match_features.csv : {len(match_feat)} matchs historiques")
    print(f"   wc2026_fixtures.csv: {len(fixtures)} matchs groupe WC2026")
    print("\nTop 10 équipes par xG moyen:")
    if "avg_xg" in team_feat.columns:
        print(team_feat[["team", "avg_xg", "fifa_rank"]].sort_values("avg_xg", ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()