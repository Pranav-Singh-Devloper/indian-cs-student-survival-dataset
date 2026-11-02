"""Score a predictions CSV on the fixed test split (honor system: labels are in the public CSV).

  python challenges/evaluate.py <task> predictions.csv
Tasks: cgpa | placed | package | burnout | persona | mess_rating
predictions.csv columns: student_id, prediction   (placed: probability of 'Yes'; others: value / label)
Only rows with split == 'test' (and a non-missing true label) are scored."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score, f1_score, mean_absolute_error, r2_score, roc_auc_score

root = Path(__file__).resolve().parent.parent
TASKS = {  # task: (label column, metric name)
    "cgpa": ("cgpa", "R2"), "placed": ("placed", "ROC-AUC"), "package": ("package_lpa", "R2 on log(package)"),
    "burnout": ("burnout_level", "macro-F1"), "persona": ("archetype", "macro-F1 (use 'cluster' for ARI)"),
    "mess_rating": ("mess_rating", "R2"),
}
task, pred_file = sys.argv[1], sys.argv[2]
col, metric = TASKS[task]
d = pd.read_csv(root / "data" / "students.csv").merge(pd.read_csv(root / "data" / "splits.csv"), on="student_id")
p = pd.read_csv(pred_file).rename(columns={"prediction": "pred"})
d = d[d.split == "test"].merge(p, on="student_id").dropna(subset=[col, "pred"])
if len(d) == 0:
    sys.exit("No overlapping test rows with labels found.")
y, yh = d[col], d.pred
if task == "placed": s = roc_auc_score((y == "Yes").astype(int), yh)
elif task == "package": s = r2_score(np.log(y), np.log(yh))
elif task in ("cgpa", "mess_rating"): s = r2_score(y, yh)
else: s = f1_score(y, yh, average="macro")
print(f"{task}: {metric} = {s:.4f}  (n={len(d)})")
