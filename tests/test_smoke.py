"""Smoke tests for ClaimsPredict — frequency-severity GLM insurance models."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import make_synthetic
from src.model import fit_and_evaluate
from src.core import poisson_deviance, gamma_deviance, tweedie_deviance, gini_index, loss_ratio


def test_data():
    """Insurance data has claim counts, amounts, and exposure."""
    d = make_synthetic(n=2000, seed=42)
    assert d["n_samples"] == 2000
    assert "claim_count" in d["df"].columns
    assert "claim_amount" in d["df"].columns
    assert "exposure" in d["df"].columns
    assert d["claim_frequency"] > 0


def test_poisson_deviance():
    """Poisson deviance is non-negative for any predictions."""
    y = np.array([0, 1, 2, 0, 3])
    p = np.array([0.5, 1.0, 1.5, 0.3, 2.0])
    dev = poisson_deviance(y, p)
    assert dev >= 0


def test_gamma_deviance():
    """Gamma deviance works on positive values."""
    y = np.array([100, 200, 300])
    p = np.array([150, 180, 250])
    dev = gamma_deviance(y, p)
    assert dev >= 0


def test_gini_index():
    """Gini index is positive when higher premiums correctly identify higher claims."""
    # Sort by premium ascending: [1,2,3,4,5] with claims [0,0,0,1,1]
    # High-claim policies have high premiums → good discrimination → positive gini
    premiums = np.array([1, 2, 3, 4, 5])
    claims = np.array([0, 0, 0, 1, 1])
    g = gini_index(premiums, claims)
    assert g > 0


def test_loss_ratio():
    """Loss ratio = claims / premiums."""
    assert abs(loss_ratio(np.array([60, 70]), np.array([100, 100])) - 0.65) < 0.01


def test_fit_and_evaluate():
    """Full frequency-severity pipeline returns model and metrics."""
    d = make_synthetic(n=2000, seed=42)
    model, metrics = fit_and_evaluate(d, seed=42)
    assert "frequency_model" in model
    assert "severity_model" in model
    assert "poisson_deviance" in metrics
    assert "gini_index" in metrics
    assert metrics["avg_pred_frequency"] > 0
    assert metrics["avg_pred_pure_premium"] > 0


if __name__ == "__main__":
    test_data()
    test_poisson_deviance()
    test_gamma_deviance()
    test_gini_index()
    test_loss_ratio()
    test_fit_and_evaluate()
    print("All ClaimsPredict smoke tests passed!")
