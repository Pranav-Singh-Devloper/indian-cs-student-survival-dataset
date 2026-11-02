"""Train on split=train, score on split=test. Produces the baseline numbers in CHALLENGES.md."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score, f1_score, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

root = Path(__file__).resolve().parent.parent
D = pd.read_csv(root / "data" / "students.csv").merge(pd.read_csv(root / "data" / "splits.csv"), on="student_id")
PLACE = ["placed", "placement_category", "offers_received", "package_lpa"]; TXT = ["mess_review_text", "reason_for_last_bunk"]
IDL = ["student_id", "archetype", "split"]

def prep(X):
    cat = X.select_dtypes(include=["object", "bool"]).columns.tolist(); num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                              ("c", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat)])

def run(name, df, y, drop, model, metric, text=None):
    tr, te = df.split == "train", df.split == "test"
    if text:
        m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2), model).fit(df.loc[tr, text], y[tr]); Xte = df.loc[te, text]
    else:
        X = df.drop(columns=drop); m = make_pipeline(prep(X), model).fit(X[tr], y[tr]); Xte = X[te]
    print(f"{name:<45}{metric(y[te], m, Xte):.3f}")

auc = lambda y, m, X: roc_auc_score(y, m.predict_proba(X)[:, 1])
r2 = lambda y, m, X: r2_score(y, m.predict(X))
f1 = lambda y, m, X: f1_score(y, m.predict(X), average="macro")
c = D.dropna(subset=["cgpa"]); run("cgpa  Ridge (R2)", c, c.cgpa, PLACE + TXT + IDL + ["cgpa", "backlogs"], Ridge(10), r2)
f = D[(D.year_of_study == 4)]; run("placed  LogReg (ROC-AUC)", f, (f.placed == "Yes").astype(int), PLACE + TXT + IDL, LogisticRegression(max_iter=2000), auc)
g = f[f.placed == "Yes"]; run("package  RF (R2 on log)", g, np.log(g.package_lpa), PLACE + TXT + IDL, RandomForestRegressor(300, n_jobs=-1, random_state=0), r2)
run("burnout  RF (macro-F1)", D, D.burnout_level, PLACE + TXT + IDL + ["burnout_level", "burnout_score"], RandomForestClassifier(300, n_jobs=-1, random_state=0), f1)
run("persona from excuse text  LogReg (macro-F1)", D, D.archetype, None, LogisticRegression(max_iter=2000), f1, text="reason_for_last_bunk")
t = D.dropna(subset=["mess_review_text"]); run("mess_rating from review  Ridge (R2)", t, t.mess_rating, None, Ridge(1.0), r2, text="mess_review_text")
cols = ["attendance_pct", "study_hours_daily", "gaming_hours_weekly", "sleep_hours", "leetcode_solved", "communication_skill",
        "club_memberships", "linkedin_connections", "certifications_count", "chai_cups_per_day", "bunked_classes_per_month", "cgpa"]
Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(np.log1p(D[cols].clip(lower=0))))
print(f"{'cluster 7 personas  KMeans (ARI, all rows)':<45}{adjusted_rand_score(D.archetype, KMeans(7, n_init=10, random_state=0).fit_predict(Z)):.3f}")
