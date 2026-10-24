# Explainable ML for Early Alzheimer's Detection (OASIS-2 Longitudinal)
Run: `pip install -r requirements.txt` then `python alzheimers_pipeline.py` (or open the notebook in Colab, upload `data/oasis_longitudinal.csv`). Runtime ~10 min on 1 CPU.
Outputs: `results/` (CV tables, rankings, versions), `figures/` (EDA, ROC, SHAP, LIME), `models/` (XGBoost explanation models).
Design: subject-wise StratifiedGroupKFold (no patient in train and test), imputation/scaling/feature selection fitted inside each fold,
two scenarios (A: with MMSE/CDR, B: MRI+demographics only), hybrid MI+RFE+SHAP feature selection, stacking ensemble vs baselines.
Data source: OASIS (Washington University ADRC) - acknowledge grants P50 AG05681, P01 AG03991, R01 AG021910, P20 MH071616, U24 RR021382.
