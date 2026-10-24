# %% [markdown]
# # Explainable Machine Learning for Early Alzheimer's Disease Detection Using Clinical and MRI-Derived Features
# **Dataset:** OASIS-2 Longitudinal (`data/oasis_longitudinal.csv`) | **Course:** Healthcare Analytics Mini Project
#
# Pipeline: EDA -> preprocessing -> two-scenario design -> hybrid feature selection -> baselines vs stacking ensemble
# -> subject-wise nested CV -> SHAP + LIME explainability -> longitudinal insight.

# %%
import os, warnings, platform
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap, joblib, sklearn, xgboost
from scipy.stats import rankdata, mannwhitneyu
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
for d in ["figures", "results", "models"]:
    os.makedirs(d, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
with open("results/versions.txt", "w") as f:
    for n, v in [("Python", platform.python_version()), ("pandas", pd.__version__), ("numpy", np.__version__),
                 ("scikit-learn", sklearn.__version__), ("xgboost", xgboost.__version__), ("shap", shap.__version__),
                 ("matplotlib", matplotlib.__version__), ("seaborn", sns.__version__)]:
        f.write(f"{n}=={v}\n")

# %% [markdown]
# ## 1. Data loading and preprocessing decisions
# - Target: `Demented` + `Converted` -> 1 (dementia at some point), `Nondemented` -> 0.
# - `Hand` is constant (all right-handed) so it is dropped. `Subject ID` is only used for grouping.
# - Missing SES (19) and MMSE (2) are imputed with the **training-fold median** inside the pipeline (no leakage).

# %%
df = pd.read_csv("data/oasis_longitudinal.csv")
print("Shape:", df.shape, "| Subjects:", df["Subject ID"].nunique(), "| Duplicates:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
df["Sex"] = (df["M/F"] == "M").astype(int)
df["y"] = (df["Group"] != "Nondemented").astype(int)
print(df["Group"].value_counts().to_string())
print("Missing:\n", df[["SES", "MMSE"]].isna().sum().to_string())

FEATS_B = ["Age", "Sex", "EDUC", "SES", "eTIV", "nWBV", "ASF"]       # Scenario B: MRI + demographics only
FEATS_A = FEATS_B + ["MMSE", "CDR"]                                  # Scenario A: + cognitive scores
y = df["y"].values
groups = df["Subject ID"].values

# outlier report (kept, not removed: values are clinically plausible)
rows = []
for c in ["Age", "EDUC", "eTIV", "nWBV", "ASF", "MMSE"]:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    rows.append((c, int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())))
print("IQR outliers:", rows)
desc = df[FEATS_A].describe().T
desc.to_csv("results/descriptive_stats.csv")

# %% [markdown]
# ## 2. Exploratory analysis with healthcare insight

# %%
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
order = ["Nondemented", "Converted", "Demented"]
pal = {"Nondemented": "#4C9F70", "Converted": "#E0A030", "Demented": "#C0392B"}
sns.countplot(data=df, x="Group", order=order, palette=pal, ax=ax[0]); ax[0].set_title("Class distribution (visits)")
for a, c in zip(ax[1:], ["nWBV", "MMSE", "Age"]):
    sns.boxplot(data=df, x="Group", y=c, order=order, palette=pal, ax=a); a.set_title(f"{c} by group")
plt.tight_layout(); plt.savefig("figures/01_eda_distribution_boxplots.png", dpi=200); plt.close()

plt.figure(figsize=(7, 6))
sns.heatmap(df[FEATS_A + ["CDR", "y"]].loc[:, ~df[FEATS_A + ["CDR", "y"]].columns.duplicated()].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation heatmap (y = dementia)"); plt.tight_layout()
plt.savefig("figures/02_correlation_heatmap.png", dpi=200); plt.close()

# subject-level longitudinal insight: yearly brain-volume (nWBV) decline
slopes = []
for sid, g in df.sort_values("MR Delay").groupby("Subject ID"):
    if len(g) >= 2 and g["MR Delay"].iloc[-1] > g["MR Delay"].iloc[0]:
        yrs = (g["MR Delay"].iloc[-1] - g["MR Delay"].iloc[0]) / 365.25
        slopes.append((sid, g["Group"].iloc[0], (g["nWBV"].iloc[-1] - g["nWBV"].iloc[0]) / yrs))
sl = pd.DataFrame(slopes, columns=["Subject", "Group", "nWBV_change_per_year"])
sl.to_csv("results/nwbv_slopes.csv", index=False)
summ = sl.groupby("Group")["nWBV_change_per_year"].agg(["count", "mean", "median"])
print(summ.to_string())
p_conv = mannwhitneyu(sl[sl.Group == "Converted"].nWBV_change_per_year,
                      sl[sl.Group == "Nondemented"].nWBV_change_per_year).pvalue
print("Converted vs Nondemented nWBV decline, Mann-Whitney p =", round(p_conv, 4))
plt.figure(figsize=(6, 4.2))
sns.boxplot(data=sl, x="Group", y="nWBV_change_per_year", order=order, palette=pal)
plt.axhline(0, color="k", lw=.7); plt.ylabel("nWBV change per year")
plt.title(f"Brain-volume decline rate (Converted vs Nondemented p={p_conv:.3f})")
plt.tight_layout(); plt.savefig("figures/03_nwbv_decline_rate.png", dpi=200); plt.close()
summ.to_csv("results/nwbv_slope_summary.csv")

# %% [markdown]
# ## 3. Proposed method: hybrid feature selection + stacking ensemble
# **HybridSelector** ranks features by three different views (Mutual Information, RFE with logistic regression,
# mean |SHAP| of a random forest), averages the ranks, and keeps the top-k. It is fitted *inside every training fold*.

# %%
class HybridSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=5, random_state=SEED):
        self.k = k; self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        r_mi = rankdata(-mutual_info_classif(X, y, random_state=self.random_state))
        r_rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=1).fit(X, y).ranking_.astype(float)
        rf = RandomForestClassifier(60, min_samples_leaf=2, random_state=self.random_state).fit(X, y)
        sv = shap.TreeExplainer(rf).shap_values(X)
        sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
        r_shap = rankdata(-np.abs(sv).mean(0))
        self.ranks_ = np.vstack([r_mi, r_rfe, r_shap])
        self.scores_ = self.ranks_.mean(0)
        self.support_ = np.sort(np.argsort(self.scores_, kind="stable")[: self.k])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


def make_models():
    rf = lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=SEED)
    xg = lambda: XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.9,
                               eval_metric="logloss", random_state=SEED, n_jobs=1)
    sv = lambda: SVC(C=1, probability=True, class_weight="balanced", random_state=SEED)
    return {
        "LogReg (baseline)": (LogisticRegression(max_iter=2000, class_weight="balanced"), {"clf__C": [0.1, 1, 10]}),
        "DecisionTree (baseline)": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                                    {"clf__max_depth": [2, 3, 5], "clf__min_samples_leaf": [5, 10]}),
        "RandomForest": (RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=SEED),
                         {"clf__max_depth": [3, None], "clf__min_samples_leaf": [2, 5]}),
        "XGBoost": (XGBClassifier(n_estimators=150, eval_metric="logloss", random_state=SEED, n_jobs=1),
                    {"clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1]}),
        "SVM": (SVC(probability=True, class_weight="balanced", random_state=SEED),
                {"clf__C": [0.5, 1, 5], "clf__gamma": ["scale", 0.1]}),
        "Stacking (proposed)": (StackingClassifier(
            [("rf", rf()), ("xgb", xg()), ("svm", sv())], final_estimator=LogisticRegression(max_iter=1000),
            cv=StratifiedKFold(3, shuffle=True, random_state=SEED), stack_method="predict_proba"), None),
    }


def build(name, use_sel, n_feats):
    clf, grid = make_models()[name]
    steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    if use_sel:
        steps.append(("sel", HybridSelector(k=int(np.ceil(0.55 * n_feats)))))
    pipe = Pipeline(steps + [("clf", clf)])
    if grid is None:
        return pipe, False
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedGroupKFold(3, shuffle=True, random_state=SEED)), True


def metrics(yt, p, thr=0.5):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    return dict(Accuracy=accuracy_score(yt, yp), Precision=precision_score(yt, yp), Sensitivity=recall_score(yt, yp),
                Specificity=tn / (tn + fp), F1=f1_score(yt, yp), AUC=roc_auc_score(yt, p))

# %% [markdown]
# ## 4. Subject-wise stratified 5-fold CV (no patient appears in both train and test)
# Run for both scenarios (A: all features, B: MRI + demographics) x (all features vs hybrid-selected).

# %%
outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
results, oof_store = [], {}
for scen, feats in [("A: Clinical+MRI", FEATS_A), ("B: MRI+Demographics (early)", FEATS_B)]:
    X = df[feats]
    for use_sel in [False, True]:
        for name in make_models():
            oof = np.zeros(len(y)); aucs = []
            for tr, te in outer.split(X, y, groups):
                assert not set(groups[tr]) & set(groups[te])
                est, is_grid = build(name, use_sel, len(feats))
                est.fit(X.iloc[tr], y[tr], **({"groups": groups[tr]} if is_grid else {}))
                oof[te] = est.predict_proba(X.iloc[te])[:, 1]
                aucs.append(roc_auc_score(y[te], oof[te]))
            m = metrics(y, oof)
            m.update(Scenario=scen, Features="Hybrid-selected" if use_sel else "All", Model=name,
                     AUC_fold_mean=np.mean(aucs), AUC_fold_std=np.std(aucs))
            results.append(m); oof_store[(scen, use_sel, name)] = oof
            print(f"{scen[:1]} | {'sel' if use_sel else 'all'} | {name:24s} AUC={m['AUC']:.3f} Sens={m['Sensitivity']:.3f} Spec={m['Specificity']:.3f}", flush=True)

res = pd.DataFrame(results)[["Scenario", "Features", "Model", "Accuracy", "Precision", "Sensitivity",
                              "Specificity", "F1", "AUC", "AUC_fold_mean", "AUC_fold_std"]].round(3)
res.to_csv("results/cv_results.csv", index=False)
with open("results/cv_results.md", "w") as f:
    f.write(res.to_markdown(index=False))
print(res.to_string())

# %% [markdown]
# ## 5. Evaluation plots: ROC, confusion matrix, scenario comparison

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, scen in zip(axes, res.Scenario.unique()):
    for name in make_models():
        sel = name.startswith("Stacking")
        key = (scen, sel, name)
        fpr, tpr, _ = roc_curve(y, oof_store[key])
        ax.plot(fpr, tpr, lw=2.6 if sel else 1.3, label=f"{name}{' + hybrid' if sel else ''} ({roc_auc_score(y, oof_store[key]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_title(f"ROC - Scenario {scen}")
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity"); ax.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig("figures/04_roc_curves.png", dpi=200); plt.close()

# scenario A vs B (all features)
sub = res[res.Features == "All"]
plt.figure(figsize=(10, 4.5))
sns.barplot(data=sub, x="Model", y="AUC", hue="Scenario"); plt.xticks(rotation=20, ha="right"); plt.ylim(0.5, 1.0)
plt.title("Effect of removing MMSE and CDR (Scenario A vs B, ROC-AUC)"); plt.tight_layout()
plt.savefig("figures/05_scenario_A_vs_B.png", dpi=200); plt.close()

# baseline vs proposed (hybrid selection effect) per scenario
plt.figure(figsize=(11, 4.5))
sns.barplot(data=res, x="Model", y="AUC", hue="Features", errorbar=None)
plt.xticks(rotation=20, ha="right"); plt.ylim(0.5, 1.0); plt.title("All features vs hybrid-selected features (mean AUC of both scenarios)")
plt.tight_layout(); plt.savefig("figures/06_feature_selection_effect.png", dpi=200); plt.close()

best = {}
for scen in res.Scenario.unique():
    s = res[res.Scenario == scen].sort_values("AUC", ascending=False).iloc[0]
    best[scen] = (s.Model, s.Features == "Hybrid-selected")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y, (oof_store[(scen, best[scen][1], best[scen][0])] >= .5).astype(int)),
                           display_labels=["Nondemented", "Demented"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Best model, scenario {scen[0]}\n{best[scen][0]} ({s.Features})", fontsize=9)
    plt.tight_layout(); plt.savefig(f"figures/07_confusion_matrix_{scen[0]}.png", dpi=200); plt.close()
print("Best per scenario:", best)

# %% [markdown]
# ## 6. Hybrid feature ranking on the full data (for reporting)

# %%
rank_tabs = []
for scen, feats in [("A", FEATS_A), ("B", FEATS_B)]:
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(df[feats]))
    hs = HybridSelector(k=int(np.ceil(0.55 * len(feats)))).fit(Xs, y)
    t = pd.DataFrame(hs.ranks_.T, index=feats, columns=["MI rank", "RFE rank", "SHAP rank"])
    t["Mean rank"] = hs.scores_; t["Selected"] = [i in hs.support_ for i in range(len(feats))]
    t.sort_values("Mean rank").to_csv(f"results/feature_ranking_{scen}.csv"); rank_tabs.append((scen, t.sort_values("Mean rank")))
    print(f"Scenario {scen}\n", t.sort_values("Mean rank").round(2).to_string())
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (scen, t) in zip(axes, rank_tabs):
    sns.heatmap(t[["MI rank", "RFE rank", "SHAP rank", "Mean rank"]], annot=True, fmt=".1f", cmap="YlGnBu_r", ax=ax, cbar=False)
    ax.set_title(f"Hybrid feature ranking - Scenario {scen} (lower = more important)")
plt.tight_layout(); plt.savefig("figures/08_hybrid_feature_ranking.png", dpi=200); plt.close()

# %% [markdown]
# ## 7. Explainable AI: SHAP (global + per-patient) and LIME
# Explanation models are XGBoost fitted on all data (explanation only; performance numbers above come from CV).

# %%
from lime.lime_tabular import LimeTabularExplainer
final = {}
for scen, feats in [("A", FEATS_A), ("B", FEATS_B)]:
    imp = SimpleImputer(strategy="median").fit(df[feats])
    Xi = pd.DataFrame(imp.transform(df[feats]), columns=feats)
    mdl = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9,
                        eval_metric="logloss", random_state=SEED, n_jobs=1).fit(Xi, y)
    joblib.dump({"imputer": imp, "model": mdl, "features": feats}, f"models/xgb_scenario_{scen}.joblib")
    final[scen] = (Xi, mdl)
    expl = shap.TreeExplainer(mdl); sv = expl(Xi)
    plt.figure(); shap.summary_plot(sv.values, Xi, show=False); plt.title(f"SHAP summary - Scenario {scen}")
    plt.tight_layout(); plt.savefig(f"figures/09_shap_summary_{scen}.png", dpi=200, bbox_inches="tight"); plt.close()
    plt.figure(); shap.summary_plot(sv.values, Xi, plot_type="bar", show=False); plt.title(f"Mean |SHAP| - Scenario {scen}")
    plt.tight_layout(); plt.savefig(f"figures/09_shap_bar_{scen}.png", dpi=200, bbox_inches="tight"); plt.close()
    # per-patient: a confidently-demented visit and a confidently-healthy visit
    pr = mdl.predict_proba(Xi)[:, 1]
    for tag, idx in [("demented", int(np.argmax(np.where(y == 1, pr, -1)))), ("nondemented", int(np.argmin(np.where(y == 0, pr, 2))))]:
        plt.figure(); shap.plots.waterfall(sv[idx], show=False)
        plt.title(f"Scenario {scen}: patient {df['Subject ID'][idx]} ({tag}, p={pr[idx]:.2f})", fontsize=9)
        plt.tight_layout(); plt.savefig(f"figures/10_shap_patient_{scen}_{tag}.png", dpi=200, bbox_inches="tight"); plt.close()

Xi, mdl = final["B"]
lime_exp = LimeTabularExplainer(Xi.values, feature_names=FEATS_B, class_names=["Nondemented", "Demented"],
                                mode="classification", random_state=SEED)
pr = mdl.predict_proba(Xi)[:, 1]
idx = int(np.argmax(np.where(y == 1, pr, -1)))
fig = lime_exp.explain_instance(Xi.values[idx], mdl.predict_proba, num_features=7).as_pyplot_figure()
fig.set_size_inches(7, 4); plt.tight_layout(); plt.savefig("figures/11_lime_patient_B.png", dpi=200); plt.close()

# SHAP vs hybrid ranking agreement + dependence of nWBV (insight)
sv_b = shap.TreeExplainer(final["B"][1])(final["B"][0])
plt.figure(figsize=(5.5, 4)); shap.dependence_plot("nWBV", sv_b.values, final["B"][0], interaction_index="Age", show=False)
plt.tight_layout(); plt.savefig("figures/12_shap_dependence_nWBV_B.png", dpi=200, bbox_inches="tight"); plt.close()
print("Done. Figures:", sorted(os.listdir("figures")))
