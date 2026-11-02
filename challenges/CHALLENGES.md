# 🏆 Challenges — beat the baselines

Fixed split: [`data/splits.csv`](../data/splits.csv) (80/20, stratified by persona, seed 2026). Train on `split == "train"`, predict `split == "test"`, and score with:

```bash
python challenges/evaluate.py <task> predictions.csv     # predictions.csv: student_id,prediction
```

> Honor system: labels are in the public CSV, so don't train on the test rows. Share your score in the Kaggle discussion or a GitHub issue/PR that edits the table below.

| # | Task | Rows | Metric | Baseline | Your move |
|---|---|---|---|---|---|
| 1 | `cgpa` — predict CGPA (drop `backlogs`) | all with CGPA | R² | **0.840** (Ridge) | Handle the MAR missingness smartly |
| 2 | `placed` — will the final-year get placed? | year 4 | ROC-AUC | **0.846** (LogReg) | Gradient boosting + feature engineering |
| 3 | `package` — predict log(package) for the placed | placed | R² | **0.420** (RF) | It's mostly company-bucket noise. Can you do better? |
| 4 | `burnout` — Low / Medium / High | all | macro-F1 | **0.683** (RF) | Don't use `burnout_score` (leak!) |
| 5 | `persona` — guess persona from the bunk excuse | all | macro-F1 | **0.760** (TF-IDF + LogReg) | Try embeddings / a small transformer |
| 6 | `mess_rating` — rating from review text | hostel & PG | R² | **0.753** (TF-IDF + Ridge) | Sentiment models |
| 7 | **Cluster** the 7 hidden personas (unsupervised) | all | ARI | **0.173** (K-Means) | GMM, UMAP+HDBSCAN, autoencoders |

Reproduce the baselines: `python challenges/baselines.py`.

## Bonus quests
* **Fairness audit:** `gender` has no causal effect by construction. Does your model still use it? Prove it with permutation importance or SHAP.
* **The semester slump:** estimate the sem 3–4 dip per persona from `semester_records.csv`.
* **Association rules:** which habits co-occur with `burnout_level == High`?
* **Causal:** does an internship *cause* a better package here? (You can read the generator to check your answer.)

## Leaderboard
| Task | Score | Who | Method |
|---|---|---|---|
| *(your name here)* | | | |
