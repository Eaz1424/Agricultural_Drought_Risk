# Predicting Agricultural Drought Risk Using Machine Learning

## Project structure

| File | Purpose |
| --- | --- |
| `predict_drought_risk.ipynb` | End-to-end workflow from data simulation to feature importances |

## Data pipeline

1. **Synthetic weather + vegetation** – Extends the Punjab monsoon generator to daily resolution (2010–2024), emitting rainfall (mm), temperature (°C), and NDVI surrogates.
2. **Feature engineering** – Builds agronomic descriptors:
   - `days_since_last_rainfall`
   - `avg_temp_past_30_days`
   - `rainfall_anomaly` vs. monthly climatology
   - `ndvi_change_rate` (7-day smoothed diffs)
   - `soil_moisture_proxy` (30-day rainfall accumulation / 30)
3. **Target labeling** – Heuristic risk score (dry spell + heat + negative anomaly + NDVI drop). Score ≥ 2 → `Drought`, otherwise `No Drought`.

## Modeling & evaluation

Models are trained with 5-fold cross-validation (scikit-learn pipelines):

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| XGBoost | 0.9987 | 0.9975 | 0.9889 | **0.9931** |
| Random Forest | 0.9985 | 0.9966 | 0.9888 | 0.9926 |
| SVM (RBF) | 0.9567 | 0.6894 | **0.9877** | 0.8110 |

Key takeaways:
- Boosted trees eke out the best F1 without sacrificing interpretability.
- The RBF SVM over-indexes recall but at the cost of precision, hinting that the synthetic classes remain imbalanced even after weighting.

## Feature importance insights

A 400-tree Random Forest highlights which signals dominate drought calls:

1. `rainfall_anomaly` – ~50% of impurity reduction; deficit conditions lead the decision boundary.
2. `avg_temp_past_30_days` & `ndvi_change_rate` – Secondary drivers, confirming that heat plus vegetation decline provides added confidence.
3. `soil_moisture_proxy` – Adds context for medium-term recharge.

## How to run

```bash
cd "/Users/eihabazizzaidi/Grad School/Project2"
jupyter notebook predict_drought_risk.ipynb
```

Run cells sequentially; the first code cell installs `xgboost` if missing and initializes styling. The notebook emits plots (NDVI vs rainfall overlays, feature importances) and tables (feature previews, CV metrics).

## Customization ideas

- Swap the synthetic generator with actual IMD rainfall + MODIS/Sentinel NDVI pulls.
- Replace the heuristic labeler with SPI/VCI thresholds or agronomic drought records.
- Introduce SHAP or permutation importances for finer-grained explanations.
- Train on 2010–2019, evaluate on 2020–2024 to mimic operational deployment.
