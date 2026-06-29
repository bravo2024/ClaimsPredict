from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_NAMES = ["policy_age_months", "claim_amount", "claimant_age", "num_prior_claims", "incident_severity", "fraud_score", "policy_type", "region", "settlement_delay_days", "legal_representation"]
CATEGORICAL_FEATURES = ["policy_type", "region", "legal_representation"]
NUMERICAL_FEATURES = ["policy_age_months", "claim_amount", "claimant_age", "num_prior_claims", "incident_severity", "fraud_score", "settlement_delay_days"]

def make_synthetic(n=10000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "policy_age_months": rng.gamma(shape=3, scale=12, size=n).clip(1, 120).astype(int),
        "claim_amount": rng.lognormal(mean=8.5, sigma=1.2, size=n).clip(500, 200000).astype(int),
        "claimant_age": rng.normal(42, 15, size=n).clip(18, 85).astype(int),
        "num_prior_claims": rng.poisson(lam=0.5, size=n).clip(0, 6),
        "incident_severity": rng.uniform(1, 10, size=n).round(2),
        "fraud_score": rng.uniform(0, 100, size=n).round(1),
        "policy_type": rng.choice(["auto", "health", "home", "life"], size=n, p=[0.35, 0.25, 0.20, 0.20]),
        "region": rng.choice(["northeast", "southeast", "midwest", "west"], size=n, p=[0.25, 0.28, 0.22, 0.25]),
        "settlement_delay_days": rng.exponential(scale=30, size=n).clip(0, 365).astype(int),
        "legal_representation": rng.choice(["yes", "no"], size=n, p=[0.35, 0.65]),
    })
    severity = df["incident_severity"] / 10; fraud = df["fraud_score"] / 100; claims = np.clip(df["num_prior_claims"], 0, 3) / 3
    delay = np.clip(df["settlement_delay_days"] / 180, 0, 1); legal = (df["legal_representation"] == "yes").astype(int)
    log_odds = -2.0 + 0.5 * severity + 0.8 * fraud + 0.3 * claims + 0.4 * delay + 0.3 * legal + rng.normal(0, 0.5, size=n)
    prob = 1 / (1 + np.exp(-log_odds))
    y = (prob > np.percentile(prob, 78)).astype(np.float64)
    return {"X": df, "y": y, "features": FEATURE_NAMES, "df": df.assign(claim_disputed=y), "categorical_features": CATEGORICAL_FEATURES, "numerical_features": NUMERICAL_FEATURES, "n_samples": n, "n_features": len(FEATURE_NAMES), "positive_rate": y.mean()}
