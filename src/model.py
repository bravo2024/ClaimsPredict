"""model.py — Frequency-severity GLM models for ClaimsPredict (Xceedance).

Implements the standard actuarial two-part model:
1. **Frequency model**: Poisson GLM with log link for claim counts,
   using exposure as an offset. Predictors: vehicle, age, territory, coverage.
2. **Severity model**: Gamma GLM with log link for claim amounts
   (conditional on at least one claim).
3. **Pure premium** = predicted frequency × predicted severity.

This is fundamentally different from generic binary classification.

References
----------
Ohlsson & Johansson (2010), "Non-life Insurance Pricing with GLMs."
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from src.core import poisson_deviance, gamma_deviance, tweedie_deviance, gini_index, loss_ratio


def _encode_features(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """One-hot encode categorical features."""
    return pd.get_dummies(df[cat_cols], drop_first=True).astype(float)


def fit_frequency_model(df: pd.DataFrame, cat_cols: list[str], exposure_col: str,
                        count_col: str) -> dict:
    """Fit Poisson GLM for claim frequency with log(exposure) offset."""
    X = _encode_features(df, cat_cols)
    X = sm.add_constant(X)
    y = df[count_col].astype(float)
    offset = np.log(df[exposure_col].astype(float))
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
    result = model.fit()
    return {"model": result, "feature_names": X.columns.tolist(), "X": X}


def fit_severity_model(df: pd.DataFrame, cat_cols: list[str], amount_col: str) -> dict:
    """Fit Gamma GLM for claim severity (positive claims only)."""
    has_claim = df[amount_col] > 0
    df_claims = df[has_claim].copy()
    if len(df_claims) < 10:
        return {"model": None, "feature_names": [], "X": None}
    X = _encode_features(df_claims, cat_cols)
    X = sm.add_constant(X)
    y = df_claims[amount_col].astype(float)
    model = sm.GLM(y, X, family=sm.families.Gamma(link=sm.families.links.log()))
    result = model.fit()
    return {"model": result, "feature_names": X.columns.tolist(), "X": X}


def fit_and_evaluate(data: dict, seed: int = 42) -> tuple:
    """Fit frequency + severity models and compute pure premium predictions.

    Returns (model_dict, metrics_dict).
    """
    df = data["df"]
    cat_cols = data["categorical_features"]
    exposure_col = data["exposure"]
    count_col = data["target_count"]
    amount_col = data["target_amount"]

    # Train/test split
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=seed)

    # Fit models
    freq_model = fit_frequency_model(df_train, cat_cols, exposure_col, count_col)
    sev_model = fit_severity_model(df_train, cat_cols, amount_col)

    # Predict on test set
    X_test = _encode_features(df_test, cat_cols)
    # Align columns with training features
    train_cols = freq_model["feature_names"]
    for col in train_cols:
        if col not in X_test.columns:
            X_test[col] = 0.0
    X_test = X_test[train_cols]
    X_test = sm.add_constant(X_test, has_constant="add")

    try:
        pred_freq = freq_model["model"].predict(X_test)
        pred_freq = np.clip(np.asarray(pred_freq, dtype=float), 0.0, None)
    except Exception:
        pred_freq = np.full(len(df_test), data.get("claim_frequency", 0.1))
    pred_sev = np.full(len(df_test), data.get("claim_severity", 3000.0))
    if sev_model["model"] is not None:
        X_test_sev = _encode_features(df_test, cat_cols)
        sev_cols = sev_model["feature_names"]
        for col in sev_cols:
            if col not in X_test_sev.columns:
                X_test_sev[col] = 1.0 if col == "const" else 0.0
        X_test_sev = X_test_sev[sev_cols]
        try:
            pred_sev = sev_model["model"].predict(X_test_sev)
            pred_sev = np.clip(pred_sev, 1.0, None)
        except Exception:
            pred_sev = np.full(len(df_test), data.get("claim_severity", 3000.0))

    pred_pure_premium = pred_freq * pred_sev
    actual_pure_premium = df_test[count_col].values * df_test[amount_col].values

    # Metrics
    metrics = {
        "n_train": len(df_train),
        "n_test": len(df_test),
        "poisson_deviance": poisson_deviance(df_test[count_col].values, pred_freq,
                                              weights=df_test[exposure_col].values),
        "gamma_deviance": gamma_deviance(df_test[df_test[amount_col] > 0][amount_col].values,
                                          pred_sev[df_test[amount_col].values > 0]),
        "tweedie_deviance": tweedie_deviance(actual_pure_premium, pred_pure_premium),
        "gini_index": gini_index(pred_pure_premium, actual_pure_premium),
        "loss_ratio": loss_ratio(actual_pure_premium, pred_pure_premium),
        "avg_pred_frequency": float(np.mean(pred_freq)),
        "avg_pred_severity": float(np.mean(pred_sev)),
        "avg_pred_pure_premium": float(np.mean(pred_pure_premium)),
    }

    model = {
        "frequency_model": freq_model,
        "severity_model": sev_model,
        "feature_names": freq_model["feature_names"],
    }
    return model, metrics
