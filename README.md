# Explainable Machine Learning for Early Alzheimer's Disease Detection Using Clinical and MRI-Derived Features

Healthcare Analytics mini project, Symbiosis Institute of Technology, Nagpur.
Authors: Sanika Jumde, Himangii Chatterjee

## Overview
This project asks how well dementia can be detected from demographic and MRI-derived features alone, compared with models that also use cognitive scores (MMSE, CDR). It uses the OASIS-2 longitudinal data and evaluates every model with subject-wise cross-validation, so no patient appears in both training and test data.

## Dataset
- **OASIS-2 longitudinal** (`oasis_longitudinal.csv`): 373 MRI visits from 150 subjects, aged 60-96.
- **Target:** Demented or Converted = 1, Nondemented = 0.
- **Source:** Open Access Series of Imaging Studies (Washington University ADRC), https://www.oasis-brains.org/. Kaggle copy: `jboysen/mri-and-alzheimers`.

## Method
1. **Preprocessing:** median imputation (SES, MMSE), standard scaling, class weights.
2. **Two scenarios:** A (all 9 features) and B (MRI + demographics only, 7 features).
3. **Hybrid feature selection:** average rank of Mutual Information, RFE and SHAP, fitted inside each training fold.
4. **Models:** Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, and a stacking ensemble (RF + XGBoost + SVM with a Logistic Regression meta-learner).
5. **Evaluation:** StratifiedGroupKFold (5 folds, grouped by subject) with inner grid-search tuning.
6. **Explainability:** SHAP (global and per-patient) and LIME.

## Main results (5-fold subject-wise CV)
| Scenario | Best model | Accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---|---|---|---|---|
| A: with MMSE and CDR | Logistic Regression | 0.941 | 0.891 | 0.989 | 0.965 |
| B: MRI + demographics only | Logistic Regression | 0.676 | 0.710 | 0.642 | 0.721 |

Key findings:
- Removing MMSE and CDR drops AUC from about 0.95 to about 0.65-0.72.
- nWBV (normalized whole-brain volume) is the most informative MRI feature.
- The stacking ensemble and hybrid feature selection did not beat Logistic Regression on this small dataset.

Full results are in `results/cv_results.csv`.

## Repository structure
```
├── alzheimers_pipeline.py          # full pipeline (run this)
├── Alzheimers_Explainable_ML.ipynb # same pipeline as a notebook
├── data/oasis_longitudinal.csv     # dataset
├── results/                        # CV results, feature rankings, versions
├── figures/                        # EDA, ROC, SHAP, LIME plots
├── models/                         # saved XGBoost explanation models
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
python alzheimers_pipeline.py
```
Runtime is about 10 minutes on one CPU core. In Google Colab, upload the notebook and place `oasis_longitudinal.csv` in a `data/` folder.

## Environment
Python 3.12, scikit-learn 1.8.0, XGBoost 3.4.1, SHAP 0.54.1, LIME 0.2.0.1, pandas 3.0.2, NumPy 2.4.4, Matplotlib 3.10.8, seaborn 0.13.2. Random seed: 42.

## Limitations
- Small dataset (150 subjects), single source, no external validation.
- All visits of a Converted subject carry the Converted label, including pre-conversion visits.
- The models are for research and are not for clinical use.

## Acknowledgement
Data were provided by OASIS (Washington University ADRC). Please acknowledge the OASIS grants (P50 AG05681, P01 AG03991, R01 AG021910, P20 MH071616, U24 RR021382) if you reuse the data.
