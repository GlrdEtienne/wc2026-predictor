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
from datetime import date
import subprocess, sys
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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .block-container { padding-top: 1rem !important; }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f2044 100%);
        border: 1px solid #1e3a5f; border-radius: 14px;
        padding: 18px 14px; box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    div[data-testid="metric-container"] label { color: #8aafd4 !important; font-size: 0.78em !important; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 1.35em !important; font-weight: 700 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #FFD700 !important; }
    .match-card {
        background: linear-gradient(135deg, #111827 0%, #1a2540 100%);
        border-radius: 12px; padding: 14px 20px; margin-bottom: 10px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.5);
    }
    h1 { color: #FFD700 !important; }
    h2, h3 { color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab-list"] {
        background: #111827; border-radius: 12px; padding: 4px 6px;
        gap: 4px; border: 1px solid #1e2d45;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 20px; font-weight: 700; color: #8aafd4;
        border-radius: 8px; padding: 14px 28px; background: transparent; border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover { background: #1e2d45; color: #e2e8f0; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #1e3a5f !important; color: #FFD700 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WC2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

GROUP_COLORS = {
    "A": "#e74c3c", "B": "#e67e22", "C": "#f39c12", "D": "#27ae60",
    "E": "#16a085", "F": "#2980b9", "G": "#8e44ad", "H": "#c0392b",
    "I": "#00b4d8", "J": "#f77f00", "K": "#52b788", "L": "#e63946",
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
    "Australia": "🇦🇺",
}

LIVE_NAME_MAP = {
    "USA": "United States",
    "Brasil": "Brazil",
    "Irak": "Iraq",
    "Curaçao": "Curacao",
}
TEAM_TO_GROUP = {t: g for g, teams in WC2026_GROUPS.items() for t in teams}


def flag(team): return FLAG_MAP.get(team, "🏳️")
def norm(name): return LIVE_NAME_MAP.get(name, name)


# ── Data loaders ──────────────────────────────────────────────────────────────
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
    return pd.read_csv(RAW / "historical/wc2026_results_live.csv", parse_dates=["date"])

@st.cache_data
def load_model_metrics():
    with open(MODELS / "model_metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_schedule():
    path = RAW / "historical/wc2026_schedule.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


# ── HTML helpers ──────────────────────────────────────────────────────────────
def match_card_mini_html(h, a, hg, ag, group, date_short):
    hw = hg > ag; aw = ag > hg
    hn = "color:#e2e8f0;font-weight:600;" if hw else ("color:#6b7280;" if aw else "color:#cbd5e1;")
    an = "color:#e2e8f0;font-weight:600;" if aw else ("color:#6b7280;" if hw else "color:#cbd5e1;")
    hs = "color:#ffffff;font-weight:800;" if hw else "color:#6b7280;"
    as_ = "color:#ffffff;font-weight:800;" if aw else "color:#6b7280;"
    ha = " ◄" if hw else ""
    aa = " ◄" if aw else ""
    return f"""
<div style="background:#161d2e;border:1px solid #1e2d45;border-radius:8px;
            padding:8px 10px;margin-bottom:5px;">
  <div style="color:#4b5e7a;font-size:0.65em;font-weight:600;letter-spacing:0.06em;margin-bottom:6px;">GROUPE {group}</div>
  <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">
    <div style="flex:1;min-width:0;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
        <span style="font-size:0.8em;{hn}">{flag(h)}&nbsp;{h}</span>
        <span style="font-size:0.85em;{hs};margin-left:6px;">{hg}{ha}</span>
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <span style="font-size:0.8em;{an}">{flag(a)}&nbsp;{a}</span>
        <span style="font-size:0.85em;{as_};margin-left:6px;">{ag}{aa}</span>
      </div>
    </div>
    <div style="border-left:1px solid #1e2d45;padding-left:8px;flex-shrink:0;text-align:center;">
      <div style="color:#38bdf8;font-size:0.62em;font-weight:700;">Termine</div>
      <div style="color:#4b5e7a;font-size:0.6em;margin-top:2px;">{date_short}</div>
    </div>
  </div>
</div>"""


def upcoming_card_html(h, a, group, fav, time_str=""):
    color = GROUP_COLORS.get(group, "#4a9eff")
    hs = "color:#e2e8f0;font-weight:600;" if fav == h else "color:#94a3b8;"
    as_ = "color:#e2e8f0;font-weight:600;" if fav == a else "color:#94a3b8;"
    time_html = f'<div style="color:#4b5e7a;font-size:0.6em;margin-top:2px;">{time_str}</div>' if time_str else ""
    return f"""
<div style="background:#161d2e;border:1px solid #1e2d45;border-radius:8px;
            padding:8px 10px;margin-bottom:5px;">
  <div style="color:#4b5e7a;font-size:0.65em;font-weight:700;letter-spacing:0.06em;margin-bottom:6px;">GROUPE {group}</div>
  <div style="display:flex;align-items:center;gap:6px;">
    <div style="flex:1;min-width:0;">
      <div style="font-size:0.8em;{hs};margin-bottom:4px;">{flag(h)}&nbsp;{h}</div>
      <div style="font-size:0.8em;{as_}">{flag(a)}&nbsp;{a}</div>
    </div>
    <div style="border-left:1px solid #1e2d45;padding-left:8px;flex-shrink:0;text-align:center;">
      <div style="color:{color};font-size:1em;font-weight:800;">vs</div>
      {time_html}
    </div>
  </div>
</div>"""


def compute_standings(group_teams, live_df):
    s = {t: {"P": 0, "W": 0, "N": 0, "D": 0, "BP": 0, "BC": 0, "Pts": 0} for t in group_teams}
    for _, m in live_df.iterrows():
        h, a = norm(m["home_team"]), norm(m["away_team"])
        if h not in s or a not in s:
            continue
        hg, ag = int(m["home_goals"]), int(m["away_goals"])
        s[h]["P"] += 1; s[h]["BP"] += hg; s[h]["BC"] += ag
        s[a]["P"] += 1; s[a]["BP"] += ag; s[a]["BC"] += hg
        if hg > ag:
            s[h]["W"] += 1; s[h]["Pts"] += 3; s[a]["D"] += 1
        elif ag > hg:
            s[a]["W"] += 1; s[a]["Pts"] += 3; s[h]["D"] += 1
        else:
            s[h]["N"] += 1; s[h]["Pts"] += 1
            s[a]["N"] += 1; s[a]["Pts"] += 1
    rows = [{"Equipe": f"{flag(t)} {t}", "J": s[t]["P"], "G": s[t]["W"],
             "N": s[t]["N"], "P": s[t]["D"], "BP": s[t]["BP"], "BC": s[t]["BC"],
             "+/-": s[t]["BP"]-s[t]["BC"], "Pts": s[t]["Pts"]} for t in group_teams]
    return pd.DataFrame(rows).sort_values(["Pts", "+/-", "BP"], ascending=False).reset_index(drop=True)


# ── Load data ─────────────────────────────────────────────────────────────────
try:
    sim      = load_simulation()
    tf       = load_team_features()
    squads   = load_squads()
    live     = load_live_results()
    metrics  = load_model_metrics()
    schedule = load_schedule()
except Exception as e:
    st.error(f"Erreur chargement donnees: {e}")
    st.stop()

rank_lookup = dict(zip(tf["team"], tf["fifa_rank"])) if "fifa_rank" in tf.columns else {}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
  <span style="font-size:3em;">🏆</span>
  <div>
    <h1 style="margin:0;font-size:2em;color:#FFD700;">FIFA World Cup 2026</h1>
    <p style="margin:0;color:#8aafd4;font-size:0.95em;">AI Predictor — XGBoost + Monte Carlo 10 000 simulations</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
top1, top2, top3 = sim.iloc[0], sim.iloc[1], sim.iloc[2]
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("🥇 Favori",      f"{flag(top1['team'])} {top1['team']}", f"{top1['prob_winner']*100:.1f}%")
with c2: st.metric("🥈 2eme favori", f"{flag(top2['team'])} {top2['team']}", f"{top2['prob_winner']*100:.1f}%")
with c3: st.metric("🥉 3eme favori", f"{flag(top3['team'])} {top3['team']}", f"{top3['prob_winner']*100:.1f}%")
with c4: st.metric("Precision modele", f"{metrics['wc_validation']['result_accuracy']*100:.1f}%", "matchs WC hist.")
with c5: st.metric("Matchs joues", str(len(live)), f"sur 72 phase groupes")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Probabilites", "⚽ Resultats live", "👥 Groupes", "📊 Stats equipes", "🔬 Modele"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Probabilités
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Probabilites de remporter la Coupe du Monde")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        top20 = sim.head(20).copy()
        top20["label"] = top20["team"].apply(lambda t: f"{flag(t)} {t}")
        fig = go.Figure()
        for col, color, name in [
            ("prob_winner", "#FFD700", "Vainqueur"),
            ("prob_final",  "#C0C0C0", "Finaliste"),
            ("prob_sf",     "#CD7F32", "Demi-finale"),
            ("prob_qf",     "#4a9eff", "Quart de finale"),
        ]:
            fig.add_trace(go.Bar(name=name, x=top20["label"], y=top20[col]*100,
                                 marker_color=color, opacity=0.88))
        fig.update_layout(
            barmode="group", template="plotly_dark", height=480,
            paper_bgcolor="#111827", plot_bgcolor="#111827",
            xaxis_tickangle=-35, yaxis_title="Probabilite (%)",
            title="Top 20 equipes — probabilites de progression",
            margin=dict(t=50, b=100),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Top 10 gagnants probables")
        for idx, (_, row) in enumerate(sim.head(10).iterrows()):
            pct = row["prob_winner"] * 100
            medal = ["🥇","🥈","🥉"][idx] if idx < 3 else f"**{idx+1}.**"
            st.markdown(f"{medal} **{flag(row['team'])} {row['team']}** — `{pct:.1f}%`")
            st.progress(min(float(row["prob_winner"]) * 10, 1.0))



# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Résultats live
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Phase de groupes")

    today = date.today()
    played_pairs = set()
    for _, m in live.iterrows():
        played_pairs.add((norm(m["home_team"]), norm(m["away_team"])))

    _, col_results, _ = st.columns([1.5, 4, 1.5])

    # ── Résultats joués ───────────────────────────────────────────────────────
    with col_results:
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            run_update = st.button("Mettre a jour", type="primary", use_container_width=True)

        if run_update:
            scores_path = Path("scores.txt")
            if not scores_path.exists():
                st.error("scores.txt introuvable a la racine du projet !")
            else:
                progress = st.progress(0, text="Lecture de scores.txt...")
                status   = st.empty()
                progress.progress(10, text="Mise a jour des resultats...")
                result_update = subprocess.run(
                    [r"C:\Users\etien\anaconda3\envs\wc2026\python.exe", "src/collect/update_live.py", "--fetch"],
                    capture_output=True, text=True,
                    env={**__import__('os').environ, "PYTHONIOENCODING": "utf-8"}
                )
                if result_update.returncode != 0:
                    st.error(f"Erreur update_live:\n{result_update.stderr}")
                    progress.empty()
                else:
                    progress.progress(30, text="Simulation Monte Carlo en cours...")
                    status.info("Simulation en cours... (~1-2 min)")
                    result_sim = subprocess.run(
                        [r"C:\Users\etien\anaconda3\envs\wc2026\python.exe", "src/model/simulate.py"],
                        capture_output=True, text=True,
                        env={**__import__('os').environ, "PYTHONIOENCODING": "utf-8"}
                    )
                    if result_sim.returncode != 0:
                        st.error(f"Erreur simulation:\n{result_sim.stderr}")
                        progress.empty(); status.empty()
                    else:
                        progress.progress(100, text="Termine !")
                        status.success("Predictions mises a jour ! Rafraichis la page.")
                        st.balloons()
                        st.cache_data.clear()

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

        if len(live) > 0:
            live_sorted = live.copy()
            live_sorted["date_key"]   = live_sorted["date"].dt.strftime("%Y-%m-%d")
            live_sorted["date_label"] = live_sorted["date"].dt.strftime("%a. %d/%m")
            live_sorted["date_short"] = live_sorted["date"].dt.strftime("%a. %d/%m")
            live_sorted = live_sorted.sort_values("date", ascending=False)

            for date_key, day_matches in live_sorted.groupby("date_key", sort=False):
                day_list   = list(day_matches.iterrows())
                date_label = day_list[0][1]["date_label"]
                st.markdown(
                    f"<div style='color:#8aafd4;font-size:0.8em;font-weight:700;"
                    f"letter-spacing:0.06em;margin:14px 0 6px;padding-bottom:5px;"
                    f"border-bottom:1px solid #1e2d45;'>Phase de groupes &nbsp;·&nbsp; {date_label}</div>",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                for i, (_, match) in enumerate(day_list):
                    h, a   = norm(match["home_team"]), norm(match["away_team"])
                    hg, ag = int(match["home_goals"]), int(match["away_goals"])
                    group  = match.get("group", TEAM_TO_GROUP.get(h, "?"))
                    with (c1 if i % 2 == 0 else c2):
                        st.markdown(match_card_mini_html(h, a, hg, ag, group, match["date_short"]), unsafe_allow_html=True)
        else:
            st.info("Aucun resultat. Ajoute des scores dans scores.txt et clique sur le bouton !")



# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Groupes
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Phase de groupes — classements & probabilites")
    group_cols = st.columns(2)

    for i, (group_name, teams) in enumerate(WC2026_GROUPS.items()):
        gc = GROUP_COLORS.get(group_name, "#4a9eff")
        with group_cols[i % 2]:
            st.markdown(
                f'<span style="background:{gc}22;color:{gc};border:1px solid {gc}44;'
                f'font-size:0.72em;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;'
                f'padding:3px 10px;border-radius:20px;display:inline-block;margin-bottom:8px;">'
                f'Groupe {group_name}</span>',
                unsafe_allow_html=True,
            )
            standings = compute_standings(teams, live)
            col_cfg = {
                "Equipe": st.column_config.TextColumn("Equipe", width="medium"),
                "J":   st.column_config.NumberColumn("J",   help="Matchs joués",            width="small"),
                "G":   st.column_config.NumberColumn("G",   help="Victoires",                width="small"),
                "N":   st.column_config.NumberColumn("N",   help="Matchs nuls",              width="small"),
                "P":   st.column_config.NumberColumn("P",   help="Défaites",                 width="small"),
                "BP":  st.column_config.NumberColumn("BP",  help="Buts pour (marqués)",      width="small"),
                "BC":  st.column_config.NumberColumn("BC",  help="Buts contre (encaissés)",  width="small"),
                "+/-": st.column_config.NumberColumn("+/-", help="Différence de buts",       width="small"),
                "Pts": st.column_config.NumberColumn("Pts", help="Points",                   width="small"),
            }
            st.dataframe(standings, column_config=col_cfg, use_container_width=True,
                         hide_index=True, height=178)

            with st.expander("Probabilites IA", expanded=False):
                gd = sim[sim["team"].isin(teams)].sort_values("prob_qualify", ascending=False)
                rows = [{"": f"{flag(r['team'])} {r['team']}",
                         "Qualif": f"{r['prob_qualify']*100:.0f}%",
                         "Titre": f"{r['prob_winner']*100:.1f}%"}
                        for _, r in gd.iterrows()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Repartition des probabilites de victoire par groupe")
    fig_sb = px.sunburst(sim, path=["group","team"], values="prob_winner",
                         color="prob_winner", color_continuous_scale="plasma",
                         template="plotly_dark")
    fig_sb.update_layout(height=580, paper_bgcolor="#111827",
                         title="Part de chaque equipe dans les probabilites de victoire")
    st.plotly_chart(fig_sb, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Stats équipes
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Stats equipes WC2026")
    _all_teams = sorted([t for g in WC2026_GROUPS.values() for t in g])
    selected_team = st.selectbox(
        "Selectionne une equipe",
        _all_teams,
        index=_all_teams.index("France"),
        format_func=lambda t: f"{flag(t)} {t}",
    )

    team_group = TEAM_TO_GROUP.get(selected_team, "?")
    gc         = GROUP_COLORS.get(team_group, "#4a9eff")
    team_sim   = sim[sim["team"] == selected_team]
    team_tf    = tf[tf["team"] == selected_team]

    col_name, col_m1, col_m2, col_m3, col_m4 = st.columns([2.5, 1, 1, 1, 1])
    with col_name:
        st.markdown(
            f'<div style="border-left:4px solid {gc};padding-left:12px;margin-top:6px;">'
            f'<span style="font-size:1.4em;font-weight:700;color:#e2e8f0;">{flag(selected_team)}&nbsp;{selected_team}</span><br>'
            f'<span style="font-size:0.75em;background:{gc}22;color:{gc};border:1px solid {gc}44;'
            f'border-radius:20px;padding:2px 10px;font-weight:700;">Groupe {team_group}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if not team_sim.empty:
        row   = team_sim.iloc[0]
        _rank = tf[tf["team"] == selected_team]["fifa_rank"].values
        _rv   = int(_rank[0]) if len(_rank) > 0 and not pd.isna(_rank[0]) else "N/A"
        col_m1.metric("Rang FIFA", _rv)
        col_m2.metric("Victoire",      f"{row['prob_winner']*100:.1f}%")
        col_m3.metric("Qualification", f"{row['prob_qualify']*100:.1f}%")
        col_m4.metric("Demi-finale",   f"{row['prob_sf']*100:.1f}%")

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    col_radar, col_bar = st.columns(2)

    with col_radar:
        if not team_tf.empty:
            radar_cols = {
                "avg_goals_per90":   "Buts/90",
                "avg_assists_per90": "Passes de/90",
                "avg_shots_per90":   "Tirs/90",
                "squad_avg_caps":    "Caps moy.",
                "fifa_points":       "Points FIFA",
            }
            avail = {k: v for k, v in radar_cols.items() if k in team_tf.columns}
            if len(avail) >= 3:
                vals, labels = [], []
                for c, label in avail.items():
                    raw = float(team_tf[c].values[0]) if not pd.isna(team_tf[c].values[0]) else 0
                    col_max = tf[c].max() if c in tf.columns and tf[c].max() > 0 else 1
                    vals.append(raw / col_max)
                    labels.append(label)
                fig_r = go.Figure(go.Scatterpolar(
                    r=vals+[vals[0]], theta=labels+[labels[0]],
                    fill="toself",
                    fillcolor=f"rgba({int(gc[1:3],16)},{int(gc[3:5],16)},{int(gc[5:7],16)},0.25)",
                    line=dict(color=gc, width=2.5),
                ))
                fig_r.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e3a5f"), bgcolor="#111827"),
                    template="plotly_dark", paper_bgcolor="#111827",
                    showlegend=False, height=320, title="Profil (normalise)",
                    margin=dict(t=40, b=20, l=30, r=30),
                )
                st.plotly_chart(fig_r, use_container_width=True)

    with col_bar:
        if team_group:
            group_teams = WC2026_GROUPS[team_group]
            comp_df = sim[sim["team"].isin(group_teams)].copy()
            comp_df["label"]     = comp_df["team"].apply(lambda t: f"{flag(t)} {t}")
            comp_df["highlight"] = comp_df["team"].apply(lambda t: gc if t == selected_team else "#334155")
            fig_b = go.Figure(go.Bar(
                x=comp_df["label"], y=comp_df["prob_winner"]*100,
                marker_color=comp_df["highlight"].tolist(),
                text=comp_df["prob_winner"].apply(lambda x: f"{x*100:.1f}%"),
                textposition="outside",
            ))
            fig_b.update_layout(
                template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#111827",
                title=f"Groupe {team_group} — prob. victoire finale",
                yaxis_title="Prob. (%)", showlegend=False, height=320,
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_b, use_container_width=True)

    # Effectif
    st.markdown("#### Effectif")
    squad = squads[squads["team"] == selected_team][["number","position","name","club","caps","age_at_wc"]]
    squad_display = squad.reset_index(drop=True)
    st.dataframe(squad_display, use_container_width=True, height=len(squad_display) * 35 + 38)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Modèle
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Metriques du modele")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Precision WC",     f"{metrics['wc_validation']['result_accuracy']*100:.1f}%")
    with c2: st.metric("MAE buts domicile", metrics['wc_validation']['home_goals_mae'])
    with c3: st.metric("MAE buts exterieur",metrics['wc_validation']['away_goals_mae'])
    with c4: st.metric("Matchs entrainement",f"{metrics['training_samples']:,}")

    st.markdown("""
    #### Architecture

    **Donnees** : FBref 2025-26 (Big 5) + resultats internationaux 2018-2026

    **Modele** : 2 XGBoost independants
    - `home_goals_model` → buts equipe domicile
    - `away_goals_model` → buts equipe exterieur

    **Features** : Ranking FIFA, Points FIFA, Forme recente (10 derniers matchs), H2H (5 derniers), Type de match

    **Simulation** : Monte Carlo 10 000 iterations — Poisson pour les scores, penalties ponderees par ranking
    """)
    st.code(json.dumps(metrics["features"], indent=2))


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center;color:#475569;font-size:0.8em;">'
    'WC2026 Predictor &nbsp;·&nbsp; Etienne Gaillard &nbsp;·&nbsp; XGBoost + Monte Carlo &nbsp;·&nbsp; FBref + Wikipedia + martj42'
    '</p>',
    unsafe_allow_html=True,
)