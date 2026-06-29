# CreditExplain

> Explainable credit risk scoring with SHAP-based model interpretability.

Trains a LightGBM classifier on synthetic credit application data and provides per-customer prediction explanations, global feature importance, and what-if analysis. Built for regulatory compliance under fair lending guidelines.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

| Metric | Value |
|---|---|
| Backend | LightGBM |
| ROC AUC | 0.849 |
| Accuracy | 0.791 |
| F1 Score | 0.688 |

## Features

| Component | What it does |
|---|---|
| **Dashboard** | Portfolio-level default rate, score distribution, approval rates by segment |
| **Borrower Explain** | Individual prediction breakdown with feature contributions |
| **What-If** | Modify borrower attributes and see updated risk score |
| **Fairness** | Adverse impact analysis across protected attributes |
| **Portfolio** | Aggregate risk metrics, concentration analysis |

## Repo Structure

```
CreditExplain/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic credit application data: income, loan amount, credit history length, employment status, DTI ratio, and behavioural features.

## License

MIT
