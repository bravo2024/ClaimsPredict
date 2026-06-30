"""core.py — Insurance claim modeling metrics for ClaimsPredict (Xceedance).

Implements the frequency-severity decomposition metrics used in
non-life insurance pricing, NOT generic classification metrics:
  * **Poisson deviance** — for claim count (frequency) models.
  * **Gamma deviance** — for claim severity models.
  * **Tweedie deviance** — for pure premium (compound Poisson-Gamma).
  * **Gini index** — insurance-specific inequality measure for premiums.
  * **Loss ratio** — claims paid / premium earned.

References
----------
Ohlsson & Johansson (2010), "Non-life Insurance Pricing with GLMs."
Dunn & Smyth (2018), "Generalized Linear Models with Examples in R."
"""
from __future__ import annotations
import numpy as np


def poisson_deviance(y_true, y_pred, weights=None) -> float:
    """Poisson deviance: 2 * sum(w * (y*log(y/yhat) - (y - yhat))).

    For claim frequency models (count data). Lower is better.
    """
    y = np.asarray(y_true, dtype=float)
    yhat = np.clip(np.asarray(y_pred, dtype=float), 1e-10, None)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    return float(2 * np.sum(w * (y * np.log(np.where(y > 0, y / yhat, 1.0)) - (y - yhat))))


def gamma_deviance(y_true, y_pred, weights=None) -> float:
    """Gamma deviance: 2 * sum(w * (-log(y/yhat) + (y - yhat)/yhat)).

    For claim severity models (positive continuous). Lower is better.
    """
    y = np.asarray(y_true, dtype=float)
    yhat = np.clip(np.asarray(y_pred, dtype=float), 1e-10, None)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    return float(2 * np.sum(w * (-np.log(y / yhat) + (y - yhat) / yhat)))


def tweedie_deviance(y_true, y_pred, p=1.5, weights=None) -> float:
    """Tweedie deviance for compound Poisson-Gamma (pure premium).

    p=1 is Poisson, p=2 is Gamma, p in (1,2) is Tweedie.
    """
    y = np.asarray(y_true, dtype=float)
    yhat = np.clip(np.asarray(y_pred, dtype=float), 1e-10, None)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if p == 1:
        return poisson_deviance(y, yhat, w)
    elif p == 2:
        return gamma_deviance(y, yhat, w)
    else:
        term1 = np.power(np.clip(y, 0, None), 2 - p) / ((1 - p) * (2 - p))
        term2 = y * np.power(yhat, 1 - p) / (1 - p)
        term3 = np.power(yhat, 2 - p) / (2 - p)
        return float(2 * np.sum(w * (term1 - term2 + term3)))


def gini_index(premiums, claims) -> float:
    """Insurance Gini index — measures how well premiums order risk.

    Sorts policies by predicted premium, computes the Lorenz curve of
    actual claims, and returns the Gini coefficient. Higher = better
    risk discrimination.
    """
    p = np.asarray(premiums, dtype=float)
    c = np.asarray(claims, dtype=float)
    if len(p) < 2 or p.sum() == 0 or c.sum() == 0:
        return 0.0
    order = np.argsort(p)
    cum_p = np.cumsum(p[order]) / p.sum()
    cum_c = np.cumsum(c[order]) / c.sum()
    # Gini = 2 * (0.5 - area_under_lorenz)
    # When high premiums correctly identify high claims, the Lorenz curve
    # of claims (sorted by premium) is below the diagonal → area < 0.5 → positive gini
    area = np.trapz(cum_c, cum_p)
    return float(2 * (0.5 - area))


def loss_ratio(claims, premiums) -> float:
    """Loss ratio = total claims / total premiums. Target ~60-70% in practice."""
    total_claims = float(np.sum(claims))
    total_premiums = float(np.sum(premiums))
    return total_claims / total_premiums if total_premiums > 0 else 0.0


def frequency_severity_validation(claim_counts, claim_amounts, exposure) -> dict:
    """Validate that frequency * severity ≈ pure premium per unit exposure."""
    freq = np.asarray(claim_counts, dtype=float)
    sev = np.asarray(claim_amounts, dtype=float)
    exp = np.asarray(exposure, dtype=float)
    mask = freq > 0
    avg_freq = freq.sum() / exp.sum() if exp.sum() > 0 else 0
    avg_sev = sev[mask].mean() if mask.any() else 0
    pure_premium = avg_freq * avg_sev
    return {
        "average_frequency": float(avg_freq),
        "average_severity": float(avg_sev),
        "pure_premium": float(pure_premium),
        "loss_ratio": loss_ratio(sev, np.full_like(sev, pure_premium * exp)) if exp.sum() > 0 else 0.0,
    }