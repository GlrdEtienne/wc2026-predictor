"""
train.py
--------
Entraîne deux modèles XGBoost :
  - home_goals_model : prédit les buts de l'équipe à domicile
  - away_goals_model : prédit les buts de l'équipe à l'extérieur

Produit :
  - models/home_goals_model.json
  - models/away_goals_model.json
  - models/feature_columns.json
  - models/model_metrics.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from loguru import logger
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

PROC   = Path("data/processed")
MODELS = Path("models")
MODELS.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")


# ── Features utilisées pour l'entraînement ────────────────────────────────────
FEATURES = [
    "home_fifa_rank",
    "away_fifa_rank",
    "fifa_rank_diff",
    "fifa_points_diff",
    "home_form_pts",
    "away_form_pts",
    "home_h2h_wins",
    "away_h2h_wins",
    "is_wc",
]

# ── Chargement ────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROC / "match_features.csv", parse_dates=["date"])

    # Filtrer : matchs avec toutes les features dispo
    df = df.dropna(subset=["home_goals", "away_goals", "home_fifa_rank", "away_fifa_rank"])

    # Garder seulement les matchs depuis 2016 (ranking FIFA plus fiable)
    df = df[df["date"] >= "2016-01-01"].copy()

    # Encoder le tournoi
    tournament_weight = {
        "FIFA World Cup": 3.0,
        "UEFA Euro": 2.0,
        "Copa América": 2.0,
        "Africa Cup of Nations": 1.5,
        "UEFA Nations League": 1.2,
        "Friendly": 0.5,
    }

    def get_weight(tournament):
        for key, w in tournament_weight.items():
            if key in str(tournament):
                return w
        return 1.0

    df["match_weight"] = df["tournament"].apply(get_weight)

    logger.info(f"Training data: {len(df)} matches ({df['date'].min().year}–{df['date'].max().year})")
    logger.info(f"WC matches: {df['is_wc'].sum()}")
    return df


# ── Entraînement ──────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame, target: str, features: list) -> tuple:
    """Entraîne un modèle XGBoost pour prédire le nombre de buts."""

    # Remplir les NaN avec la médiane
    X = df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    y = df[target].clip(0, 10)  # cap à 10 buts max
    w = df["match_weight"]

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mae",
        early_stopping_rounds=20,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logger.info(f"  {target}: MAE={mae:.3f}, RMSE={rmse:.3f}")

    # Feature importance
    importance = dict(zip(features, model.feature_importances_))
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"  Top features: {top}")

    return model, {"mae": round(mae, 4), "rmse": round(rmse, 4)}


# ── Validation sur matchs WC uniquement ──────────────────────────────────────
def validate_on_wc(df: pd.DataFrame, home_model, away_model, features: list):
    """Valide les prédictions sur les matchs WC historiques."""
    wc_df = df[df["is_wc"] == 1].copy()

    X = wc_df[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())

    wc_df["pred_home"] = home_model.predict(X).clip(0, 10)
    wc_df["pred_away"] = away_model.predict(X).clip(0, 10)

    # Résultat prédit vs réel
    wc_df["pred_result"] = np.where(
        wc_df["pred_home"] > wc_df["pred_away"], 1,
        np.where(wc_df["pred_home"] < wc_df["pred_away"], -1, 0)
    )

    accuracy = (wc_df["pred_result"] == wc_df["result"]).mean()
    logger.info(f"\nWC-only validation ({len(wc_df)} matches):")
    logger.info(f"  Result accuracy: {accuracy:.1%}")

    # MAE sur les buts
    home_mae = mean_absolute_error(wc_df["home_goals"], wc_df["pred_home"])
    away_mae = mean_absolute_error(wc_df["away_goals"], wc_df["pred_away"])
    logger.info(f"  Home goals MAE: {home_mae:.3f}")
    logger.info(f"  Away goals MAE: {away_mae:.3f}")

    return accuracy, home_mae, away_mae


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logger.info("=== Training Models ===")

    df = load_data()

    logger.info(f"\nFeatures used ({len(FEATURES)}): {FEATURES}")

    # Entraîner les deux modèles
    logger.info("\nTraining home goals model...")
    home_model, home_metrics = train_model(df, "home_goals", FEATURES)

    logger.info("\nTraining away goals model...")
    away_model, away_metrics = train_model(df, "away_goals", FEATURES)

    # Validation WC
    acc, h_mae, a_mae = validate_on_wc(df, home_model, away_model, FEATURES)

    # Sauvegarder les modèles
    home_model.save_model(str(MODELS / "home_goals_model.json"))
    away_model.save_model(str(MODELS / "away_goals_model.json"))

    # Sauvegarder les métadonnées
    metadata = {
        "features": FEATURES,
        "home_model": home_metrics,
        "away_model": away_metrics,
        "wc_validation": {
            "result_accuracy": round(acc, 4),
            "home_goals_mae": round(h_mae, 4),
            "away_goals_mae": round(a_mae, 4),
        },
        "training_samples": len(df),
    }
    with open(MODELS / "model_metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(MODELS / "feature_columns.json", "w") as f:
        json.dump(FEATURES, f)

    logger.success(f"\nModels saved to {MODELS}/")
    print(f"""
✅ Training complete!
   Home goals MAE : {home_metrics['mae']}
   Away goals MAE : {away_metrics['mae']}
   WC result accuracy : {acc:.1%}
    """)


if __name__ == "__main__":
    main()