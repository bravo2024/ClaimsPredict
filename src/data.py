"""data.py — Synthetic insurance policy data for ClaimsPredict (Xceedance).

Policy-level data with exposure, claim count, claim amount, and rating
factors (vehicle type, driver age, territory, coverage). This mirrors
the structure of real auto insurance data used in GLM pricing.

The frequency-severity decomposition is the standard actuarial approach:
  * Claim **frequency** (count per unit exposure) → Poisson GLM
  * Claim **severity** (average amount per claim) → Gamma GLM
  * **Pure premium** = frequency × severity
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any


def make_synthetic(n: int = 5000, seed: int = 42) -> dict[str, Any]:
    """Generate synthetic auto insurance policy data.

    Rating factors: vehicle_type, driver_age_group, territory, coverage_level.
    Frequency depends on driver age and territory. Severity depends on
    vehicle type and coverage level.
    """
    rng = np.random.default_rng(seed)

    vehicle_type = rng.choice(["sedan", "suv", "truck", "sports"], n, p=[0.35, 0.30, 0.20, 0.15])
    driver_age = rng.choice(["18-25", "26-40", "41-60", "60+"], n, p=[0.15, 0.35, 0.35, 0.15])
    territory = rng.choice(["urban", "suburban", "rural"], n, p=[0.40, 0.40, 0.20])
    coverage = rng.choice(["basic", "standard", "premium"], n, p=[0.25, 0.50, 0.25])
    exposure = rng.uniform(0.5, 1.0, n).round(3)

    # Frequency: young + urban = more claims, rural = fewer
    base_freq = np.full(n, 0.10)
    age_effect = {"18-25": 1.8, "26-40": 1.0, "41-60": 0.7, "60+": 0.9}
    terr_effect = {"urban": 1.5, "suburban": 1.0, "rural": 0.6}
    freq_rate = base_freq * np.array([age_effect[a] for a in driver_age]) * \
                np.array([terr_effect[t] for t in territory])
    claim_count = rng.poisson(freq_rate * exposure).astype(int)

    # Severity: sports + premium = higher claims
    base_sev = 3000
    veh_effect = {"sedan": 1.0, "suv": 1.2, "truck": 1.1, "sports": 1.8}
    cov_effect = {"basic": 0.7, "standard": 1.0, "premium": 1.5}
    sev_rate = base_sev * np.array([veh_effect[v] for v in vehicle_type]) * \
               np.array([cov_effect[c] for c in coverage])
    claim_amount = np.zeros(n)
    has_claim = claim_count > 0
    claim_amount[has_claim] = rng.gamma(
        shape=3.0, scale=sev_rate[has_claim] / 3.0
    ).round(2)

    df = pd.DataFrame({
        "vehicle_type": vehicle_type, "driver_age_group": driver_age,
        "territory": territory, "coverage_level": coverage,
        "exposure": exposure, "claim_count": claim_count,
        "claim_amount": claim_amount,
    })

    return {
        "df": df,
        "features": ["vehicle_type", "driver_age_group", "territory", "coverage_level"],
        "categorical_features": ["vehicle_type", "driver_age_group", "territory", "coverage_level"],
        "numerical_features": ["exposure"],
        "target_count": "claim_count",
        "target_amount": "claim_amount",
        "exposure": "exposure",
        "n_samples": n,
        "claim_frequency": float(claim_count.sum() / exposure.sum()),
        "claim_severity": float(claim_amount[has_claim].mean()) if has_claim.any() else 0.0,
        "loss_ratio": float(claim_amount.sum() / (exposure * 1000).sum()),  # rough premium proxy
    }