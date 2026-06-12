"""
app.py — WC2026 Predictor Dashboard
streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC2026 Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC   = Path("data/processed")
MODELS = Path("models")
RAW    = Path("data/raw")

# ── CSS custom ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .team-winner {
        font-size: 1.4em;
        font-weight: bold;
        color: #FFD700;
    }
    .prob-bar { height: 8px; border-radius: 4px; background: #1a1a2e; }
    h1 { color: #FFD700 !important; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_simulation():
    return pd.read_csv(PROC / "simulation_results.csv")

@st.cache_data
def load_team_features():
    return pd.read_csv(PROC / "team_features.csv")

@st.cache_data
def load_squads():
    return pd.read_csv(RAW / "squads/wc2026_squads.csv")

@st.cache_data
def load_live_results():
    df = pd.read_csv(RAW / "historical/wc2026_results_live.csv", parse_dates=["date"])
    return df

@st.cache_data
def load_model_metrics():
    with open(MODELS / "model_metrics.json") as f:
        return json.load(f)

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

FLAG_MAP = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czech Republic": "🇨🇿",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    "Canada": "🇨🇦", "Bosnia and Herzegovina": "🇧🇦", "Ukraine": "🇺🇦", "Jordan": "🇯🇴",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🇬🇧",
    "Germany": "🇩🇪", "Ecuador": "🇪🇨", "Ivory Coast": "🇨🇮", "Curacao": "🇨🇼",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Tunisia": "🇹🇳", "Sweden": "🇸🇪",
    "Spain": "🇪🇸", "Cape Verde": "🇨🇻", "Uzbekistan": "🇺🇿", "DR Congo": "🇨🇩",
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Colombia": "🇨🇴", "Algeria": "🇩🇿",
    "Argentina": "🇦🇷", "Austria": "🇦🇹", "Panama": "🇵🇦", "Ghana": "🇬🇭",
    "Portugal": "🇵🇹", "Croatia": "🇭🇷", "Turkey": "🇹🇷", "England": "🇬🇧",
}


def flag(team):
    return FLAG_MAP.get(team, "🏳️")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏆 FIFA World Cup 2026 — AI Predictor")
st.caption("Modèle XGBoost + Monte Carlo 10 000 simulations | Données FBref 2025-26")

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    sim    = load_simulation()
    tf     = load_team_features()
    squads = load_squads()
    live   = load_live_results()
    metrics = load_model_metrics()
    data_ok = True
except Exception as e:
    st.error(f"Erreur chargement données: {e}")
    data_ok = False
    st.stop()

# ── KPIs top ──────────────────────────────────────────────────────────────────
top1 = sim.iloc[0]
top2 = sim.iloc[1]
top3 = sim.iloc[2]

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("🥇 Favori", f"{flag(top1['team'])} {top1['team']}", f"{top1['prob_winner']*100:.1f}%")
with c2:
    st.metric("🥈 2ème favori", f"{flag(top2['team'])} {top2['team']}", f"{top2['prob_winner']*100:.1f}%")
with c3:
    st.metric("🥉 3ème favori", f"{flag(top3['team'])} {top3['team']}", f"{top3['prob_winner']*100:.1f}%")
with c4:
    st.metric("🎯 Précision modèle", f"{metrics['wc_validation']['result_accuracy']*100:.1f}%", "sur matchs WC hist.")
with c5:
    st.metric("📊 Matchs simulés", "10 000", "Monte Carlo")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Probabilités", "👥 Groupes", "⚽ Résultats live", "📊 Stats équipes", "🔬 Modèle"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Probabilités de victoire
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Probabilités de remporter la Coupe du Monde")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Bar chart top 20
        top20 = sim.head(20).copy()
        top20["label"] = top20["team"].apply(lambda t: f"{flag(t)} {t}")

        fig = go.Figure()
        stages = [
            ("prob_winner", "#FFD700", "🏆 Vainqueur"),
            ("prob_final",  "#C0C0C0", "Finaliste"),
            ("prob_sf",     "#CD7F32", "Demi-finale"),
            ("prob_qf",     "#4a9eff", "Quart de finale"),
        ]

        for col, color, name in stages:
            fig.add_trace(go.Bar(
                name=name,
                x=top20["label"],
                y=top20[col] * 100,
                marker_color=color,
                opacity=0.85,
            ))

        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            height=480,
            showlegend=True,
            xaxis_tickangle=-35,
            yaxis_title="Probabilité (%)",
            title="Top 20 équipes — probabilités de progression",
            margin=dict(t=50, b=100),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🏆 Top 10 gagnants probables")
        for _, row in sim.head(10).iterrows():
            pct = row["prob_winner"] * 100
            st.markdown(
                f"**{flag(row['team'])} {row['team']}** — {pct:.1f}%"
            )
            st.progress(min(float(row["prob_winner"]) * 10, 1.0))  # scale for visibility

    # Tableau complet
    st.subheader("Tableau complet des probabilités")
    display_df = sim.copy()
    display_df["team"] = display_df["team"].apply(lambda t: f"{flag(t)} {t}")
    for col in ["prob_winner", "prob_final", "prob_sf", "prob_qf", "prob_r32", "prob_qualify"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.1f}%")
    display_df.columns = ["Équipe", "Groupe", "Vainqueur", "Finale", "½ Finale", "¼ Finale", "R32", "Phase de groupes"]
    st.dataframe(display_df, use_container_width=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Groupes
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Phase de groupes — probabilités de qualification")

    group_cols = st.columns(3)
    groups_list = list(WC2026_GROUPS.items())

    for i, (group_name, teams) in enumerate(groups_list):
        col = group_cols[i % 3]
        with col:
            st.markdown(f"### Groupe {group_name}")
            group_data = sim[sim["team"].isin(teams)].copy()
            group_data = group_data.sort_values("prob_qualify", ascending=False)

            rows = []
            for _, row in group_data.iterrows():
                rows.append({
                    "": f"{flag(row['team'])} {row['team']}",
                    "Qualif%": f"{row['prob_qualify']*100:.0f}%",
                    "Victoire%": f"{row['prob_winner']*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Sunburst chart par groupe
    st.subheader("Répartition des probabilités de victoire par groupe")
    fig_sb = px.sunburst(
        sim,
        path=["group", "team"],
        values="prob_winner",
        color="prob_winner",
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig_sb.update_layout(height=600, title="Part de chaque équipe dans les probabilités de victoire")
    st.plotly_chart(fig_sb, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Résultats live + bouton update
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("⚽ Résultats WC2026 — Phase de groupes")

    # ── Bouton Update ─────────────────────────────────────────────────────────
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_update = st.button("🔄 Mettre à jour les prédictions", type="primary", use_container_width=True)
    with col_info:
        st.caption("Lance la lecture de `scores.txt` + re-simulation Monte Carlo (10 000 itérations)")

    if run_update:
        scores_path = Path("scores.txt")
        if not scores_path.exists():
            st.error("❌ `scores.txt` introuvable à la racine du projet !")
        else:
            # Progress bar
            progress = st.progress(0, text="⏳ Lecture de scores.txt...")
            status   = st.empty()

            import subprocess, sys, time

            # Étape 1 — update_live (fetch + patch simulate.py)
            progress.progress(10, text="📥 Mise à jour des résultats...")
            result_update = subprocess.run(
                [sys.executable, "src/collect/update_live.py", "--fetch"],
                capture_output=True, text=True
            )
            if result_update.returncode != 0:
                st.error(f"❌ Erreur update_live:\n```\n{result_update.stderr}\n```")
                progress.empty()
            else:
                progress.progress(30, text="🤖 Lancement de la simulation Monte Carlo (10 000 itérations)...")
                status.info("⚙️ Simulation en cours... (~1-2 min)")

                # Étape 2 — simulate
                result_sim = subprocess.run(
                    [sys.executable, "src/model/simulate.py"],
                    capture_output=True, text=True
                )

                if result_sim.returncode != 0:
                    st.error(f"❌ Erreur simulation:\n```\n{result_sim.stderr}\n```")
                    progress.empty()
                    status.empty()
                else:
                    progress.progress(100, text="✅ Simulation terminée !")
                    status.success("✅ Prédictions mises à jour ! Rafraîchis la page pour voir les nouveaux résultats.")
                    st.balloons()
                    # Clear cache pour recharger les données
                    st.cache_data.clear()

    st.divider()

    # ── Résultats affichés ────────────────────────────────────────────────────
    if len(live) > 0:
        for _, match in live.iterrows():
            h, a = match["home_team"], match["away_team"]
            hg, ag = int(match["home_goals"]), int(match["away_goals"])
            group = match.get("group", "?")

            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.markdown(f"### {flag(h)} {h}")
            with col2:
                color = "#FFD700" if hg > ag else ("#FF4444" if hg < ag else "#888")
                st.markdown(
                    f"<h2 style='text-align:center; color:{color}'>{hg} — {ag}</h2>",
                    unsafe_allow_html=True
                )
                st.caption(f"Groupe {group} | {str(match['date'])[:10]}")
            with col3:
                st.markdown(f"### {flag(a)} {a}")
            st.divider()
    else:
        st.info("Aucun résultat enregistré. Ajoute des scores dans `scores.txt` et clique sur le bouton !")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Stats équipes
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 Stats équipes WC2026")

    selected_team = st.selectbox(
        "Sélectionne une équipe",
        sorted([t for group in WC2026_GROUPS.values() for t in group]),
        format_func=lambda t: f"{flag(t)} {t}"
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"#### {flag(selected_team)} {selected_team}")

        # Stats simulation
        team_sim = sim[sim["team"] == selected_team]
        if not team_sim.empty:
            row = team_sim.iloc[0]
            _rank = tf[tf["team"] == selected_team]["fifa_rank"].values
            _rank_val = int(_rank[0]) if len(_rank) > 0 and not pd.isna(_rank[0]) else "N/A"
            st.metric("Rang mondial (FIFA)", _rank_val)
            st.metric("Probabilité victoire finale", f"{row['prob_winner']*100:.1f}%")
            st.metric("Probabilité qualification", f"{row['prob_qualify']*100:.1f}%")
            st.metric("Probabilité demi-finale", f"{row['prob_sf']*100:.1f}%")

        # Squad
        st.markdown("#### Effectif")
        squad = squads[squads["team"] == selected_team][["number", "position", "name", "club", "caps", "age_at_wc"]]
        st.dataframe(squad.reset_index(drop=True), use_container_width=True, height=300)

    with col_b:
        # Radar chart des stats
        team_tf = tf[tf["team"] == selected_team]
        if not team_tf.empty:
            radar_cols = {
                "avg_goals_per90":    "Buts/90",
                "avg_assists_per90":  "Passes dé/90",
                "avg_shots_per90":    "Tirs/90",
                "squad_avg_caps":     "Caps moy.",
                "fifa_points":        "Points FIFA",
            }
            available_radar = {k: v for k, v in radar_cols.items() if k in team_tf.columns}

            if len(available_radar) >= 3:
                vals = []
                labels = []
                for col, label in available_radar.items():
                    raw = float(team_tf[col].values[0]) if not pd.isna(team_tf[col].values[0]) else 0
                    # Normaliser 0-1 par rapport au max global
                    col_max = tf[col].max() if col in tf.columns else 1
                    vals.append(raw / col_max if col_max > 0 else 0)
                    labels.append(label)

                fig_radar = go.Figure(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=labels + [labels[0]],
                    fill="toself",
                    fillcolor="rgba(74, 158, 255, 0.3)",
                    line=dict(color="#4a9eff", width=2),
                    name=selected_team,
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    template="plotly_dark",
                    showlegend=False,
                    height=350,
                    title=f"Profil de {selected_team} (normalisé)",
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        # Comparaison avec d'autres équipes
        st.markdown("#### Comparaison — probabilités de victoire")
        group_of_team = next((g for g, teams in WC2026_GROUPS.items() if selected_team in teams), None)
        if group_of_team:
            group_teams = WC2026_GROUPS[group_of_team]
            comp_df = sim[sim["team"].isin(group_teams)].copy()
            comp_df["label"] = comp_df["team"].apply(lambda t: f"{flag(t)} {t}")
            fig_bar = px.bar(
                comp_df,
                x="label",
                y="prob_winner",
                color="prob_winner",
                color_continuous_scale="Blues",
                template="plotly_dark",
                title=f"Groupe {group_of_team} — probabilités de victoire finale",
                labels={"prob_winner": "Prob. victoire", "label": ""},
            )
            fig_bar.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Modèle
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔬 Métriques du modèle")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Précision résultats WC", f"{metrics['wc_validation']['result_accuracy']*100:.1f}%")
    with c2:
        st.metric("MAE buts domicile", metrics['wc_validation']['home_goals_mae'])
    with c3:
        st.metric("MAE buts extérieur", metrics['wc_validation']['away_goals_mae'])
    with c4:
        st.metric("Matchs d'entraînement", f"{metrics['training_samples']:,}")

    st.markdown("""
    #### Architecture du modèle

    **Données** : FBref 2025-26 (Big 5 européens) + 15 819 résultats internationaux 2010-2026

    **Modèle** : Deux XGBoost indépendants
    - `home_goals_model` → prédit les buts de l'équipe 1
    - `away_goals_model` → prédit les buts de l'équipe 2

    **Features** :
    - Ranking FIFA (home + away + différence)
    - Points FIFA (différence)
    - Forme récente (points sur 10 derniers matchs)
    - H2H (victoires sur 5 derniers face-à-face)
    - Type de match (WC vs amical)

    **Simulation** : Monte Carlo 10 000 itérations
    - Phase de groupes simulée via distribution de Poisson
    - Phase éliminatoire avec penalties (probabilité pondérée par ranking)
    - Résultats déjà joués fixés, matchs futurs simulés
    """)

    st.markdown("#### Features utilisées")
    st.code(json.dumps(metrics["features"], indent=2))


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("WC2026 Predictor | Etienne Gaillard | XGBoost + Monte Carlo | Données FBref + Wikipedia + martj42")