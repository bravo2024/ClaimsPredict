# ClaimsPredict

> P&C insurance claim severity prediction with chain ladder reserving.

Trains four classifiers on synthetic claims data to predict high-severity claim outcomes. The dashboard provides a full actuarial toolkit: chain ladder development triangles, loss distribution modelling, Solvency II reserve estimation, and what-if scenario analysis.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.657 |
| Gini | 0.314 |
| KS Statistic | 0.331 |
| F1 Score | 0.415 |
| Accuracy | 0.616 |

5-fold CV AUC: 0.658 ± 0.032. Four models compared.

## Features

| Component | What it does |
|---|---|
| **Controls** | Policy type filter, development periods, confidence level, expense load, claim threshold |
| **Loss Triangle** | Chain ladder development triangles with link ratio estimation |
| **Severity Model** | Multi-model classification results, ROC curves, calibration |
| **Reserving** | IBNR reserve estimates, cash flow projection, confidence intervals |
| **Scenario** | What-if stress testing on loss development assumptions |

## Repo Structure

```
ClaimsPredict/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic P&C claims portfolio: policy type (auto/home/commercial), accident severity, driver/vehicle attributes, weather conditions, prior claims history, coverage amounts. 15,000 claim records with ~30% zero-payout.

## License

MIT
