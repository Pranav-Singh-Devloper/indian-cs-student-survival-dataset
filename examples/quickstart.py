"""Baselines across classic ML tasks on the Desi CS Student Survival Dataset.
Run from repo root:  python examples/quickstart.py
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
D = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "students.csv")
RS = 42

# Columns that would leak the answer for each task
PLACEMENT_COLS = ["placed", "placement_category", "offers_received", "package_lpa"]
TEXT_COLS = ["mess_review_text", "reason_for_last_bunk"]
ID_LABEL = ["student_id", "archetype"]


def prep(df):
    cat = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    num = [c for c in df.columns if c not in cat]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat),
    ])


def report(name, value):
    print(f"{name:<58}{value}")


# 1. Regression: predict CGPA ------------------------------------------------
r = D.dropna(subset=["cgpa"]).drop(columns=PLACEMENT_COLS + TEXT_COLS + ID_LABEL + ["backlogs"])
X, y = r.drop(columns="cgpa"), r["cgpa"]
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=RS)
for nm, m in [("Linear Regression", LinearRegression()), ("Ridge", Ridge(10)),
              ("Random Forest", RandomForestRegressor(200, n_jobs=-1, random_state=RS))]:
    p = make_pipeline(prep(X), m).fit(Xtr, ytr)
    report(f"[Regression] CGPA  / {nm}  (R2)", round(r2_score(yte, p.predict(Xte)), 3))

# 2. Binary classification: placed? (final-year only) -------------------------
f = D[D.year_of_study == 4].copy()
y = (f.placed == "Yes").astype(int)
X = f.drop(columns=PLACEMENT_COLS + TEXT_COLS + ID_LABEL)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=RS, stratify=y)
for nm, m in [("Logistic Regression", LogisticRegression(max_iter=1000)),
              ("Random Forest", RandomForestClassifier(300, n_jobs=-1, random_state=RS))]:
    p = make_pipeline(prep(X), m).fit(Xtr, ytr)
    report(f"[Classification] Placed? / {nm}  (ROC-AUC)", round(roc_auc_score(yte, p.predict_proba(Xte)[:, 1]), 3))

# 3. Regression on a skewed target: package among the placed -----------------
g = f[f.placed == "Yes"]
X = g.drop(columns=PLACEMENT_COLS + TEXT_COLS + ID_LABEL)
y = np.log(g.package_lpa)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=RS)
p = make_pipeline(prep(X), RandomForestRegressor(300, n_jobs=-1, random_state=RS)).fit(Xtr, ytr)
report("[Regression] log(package_lpa) / Random Forest  (R2)", round(r2_score(yte, p.predict(Xte)), 3))

# 4. Multiclass: burnout level ------------------------------------------------
X = D.drop(columns=["burnout_level", "burnout_score"] + PLACEMENT_COLS + TEXT_COLS + ID_LABEL)
y = D.burnout_level
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=RS, stratify=y)
p = make_pipeline(prep(X), RandomForestClassifier(300, n_jobs=-1, random_state=RS)).fit(Xtr, ytr)
report("[Multiclass] Burnout level / Random Forest  (macro-F1)",
       round(f1_score(yte, p.predict(Xte), average="macro"), 3))

# 5. Clustering: can K-Means rediscover the 7 hidden archetypes? --------------
num = D[["attendance_pct", "study_hours_daily", "gaming_hours_weekly", "sleep_hours", "leetcode_solved",
         "communication_skill", "club_memberships", "linkedin_connections", "certifications_count",
         "chai_cups_per_day", "bunked_classes_per_month", "cgpa"]]
Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(np.log1p(num.clip(lower=0))))
km = KMeans(7, n_init=10, random_state=RS).fit(Z)
report("[Clustering] KMeans(k=7) vs hidden archetype  (ARI)", round(adjusted_rand_score(D.archetype, km.labels_), 3))

# 6. PCA ----------------------------------------------------------------------
pca = PCA().fit(Z)
report("[PCA] components for 90% variance (of 12 behaviour features)",
       int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1))

# 7. NLP: sentiment of mess reviews -> mess_rating -----------------------------
t = D.dropna(subset=["mess_review_text"])
Xtr, Xte, ytr, yte = train_test_split(t.mess_review_text, t.mess_rating, random_state=RS)
p = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), Ridge(1.0)).fit(Xtr, ytr)
report("[NLP] TF-IDF + Ridge: mess_review_text -> mess_rating  (R2)", round(r2_score(yte, p.predict(Xte)), 3))

# 8. NLP: bunk excuse -> archetype ---------------------------------------------
Xtr, Xte, ytr, yte = train_test_split(D.reason_for_last_bunk, D.archetype, random_state=RS, stratify=D.archetype)
p = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
report("[NLP] excuse text -> archetype / TF-IDF + LogReg  (accuracy)", round(accuracy_score(yte, p.predict(Xte)), 3))
