# 🎒 Desi CS Student Survival Dataset

[![License: CC0](https://img.shields.io/badge/license-CC0--1.0-blue)](LICENSE) [![Kaggle](https://img.shields.io/badge/Kaggle-dataset-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/pranavsinghdeveloper/desi-cs-student-survival) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-dataset-yellow)](https://huggingface.co/datasets/pranav-singh-developer-1/desi-cs-student-survival) [![Demo](https://img.shields.io/badge/demo-Will%20I%20get%20placed%3F-orange)](https://huggingface.co/spaces/pranav-singh-developer-1/will-i-get-placed) ![Synthetic](https://img.shields.io/badge/data-100%25%20synthetic-success)

> *A synthetic, funny, quirky and painfully relatable dataset about life as an Indian Computer Science student — hostel, mess, bunking, backlogs, LeetCode guilt, CGPA anxiety and placement season.*

**10,000 students · 59 features · 7 hidden personas · 8 semesters of history · free-text mess reviews & bunk excuses**

Built so you can practise **almost the entire classical ML syllabus** on data that doesn't make you want to fall asleep (unlike Iris, which hasn't bunked a single class).

> ⚠️ **100% synthetic.** No real student, college or company appears here. The data comes from a hand-written causal story ([`src/generate_dataset.py`](src/generate_dataset.py)), so patterns are realistic-*ish*, not research-grade. Don't use it to draw conclusions about real people.

## What's inside

| File | Rows | What |
|---|---|---|
| [`data/students.csv`](data/students.csv) | 10,000 | One row per student: background, habits, mess, skills, mental health, placement |
| [`data/semester_records.csv`](data/semester_records.csv) | 48,876 | Long-format SGPA / backlogs / attendance per semester |
| [`data/splits.csv`](data/splits.csv) | 10,000 | Fixed 80/20 train/test split for fair comparison |
| [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) | | Every column explained + leakage warnings |
| [`src/generate_dataset.py`](src/generate_dataset.py) | | Seeded generator (`seed=42`), fully reproducible |
| [`examples/quickstart.py`](examples/quickstart.py) | | Baselines for every task below |
| [`challenges/`](challenges/CHALLENGES.md) | | 7 challenges + evaluator + leaderboard 🏆 |
| [`app/`](app/app.py) | | Streamlit demo: *Will I get placed?* |
| [`notebooks/kaggle/`](notebooks/kaggle) | | 5 notebooks: quickstart, placement, personas, NLP, time series |

## The story behind the numbers

Every student is one of 7 archetypes — 📚 `library_topper`, 🛌 `backbench_legend`, 💻 `leetcode_grinder`, 🎉 `hostel_socialite`, 📜 `certificate_collector`, 🎮 `gaming_ghost`, 📣 `club_netaji` — which sets hidden traits (aptitude, discipline, coding passion, sociability, chaos). Those drive habits → semester grades → CGPA & backlogs → skills → stress/burnout → placement outcome and package. Built-in truths:

* Sem 3–4 has a **DSA/OS/DBMS slump**; senior year has **senioritis**.
* **Hostel wifi < PG wifi.** Maggi consumption: hostel ≫ day scholar.
* Placement depends on CGPA, LeetCode, internships, backlogs, communication, college tier (+ noise). Package is right-skewed by company type.
* `gender` has **no effect** on any outcome (handy for fairness checks).
* Real-life mess: MCAR missing values, MAR missing CGPA (low scorers "forget" to report), ~1% GitHub-commit outliers, class imbalance.

## Practice menu

| Technique | Task | Target / columns |
|---|---|---|
| **EDA, imputation, outliers** | Handle missingness (MCAR/MAR/structural), skew | `cgpa`, `github_commits_last_year`, `mess_rating` |
| **Linear / Ridge / Lasso regression** | Predict CGPA | `cgpa` |
| **Polynomial / log transform** | Skewed targets | `package_lpa`, `leetcode_solved` |
| **Logistic regression, SVM, kNN, trees, RF, boosting** | Will they get placed? | `placed` (final-years) |
| **Multiclass + imbalance (SMOTE, class weights)** | Placement bucket | `placement_category` |
| **Multiclass** | Burnout level | `burnout_level` |
| **K-Means, GMM, hierarchical, DBSCAN** | Rediscover the 7 personas | `archetype` is the answer key |
| **PCA / t-SNE / UMAP** | Visualise habits | behaviour columns |
| **Naive Bayes, TF-IDF, sentiment** | Mess review → rating; excuse → persona | `mess_review_text`, `reason_for_last_bunk` |
| **Association rules** | "Maggi + chai + all-nighter →?" | binarised habits |
| **Time series / panel** | Semester slump, trend per student | `semester_records.csv` |
| **Multicollinearity / feature selection** | `study_hours` vs `attendance` vs `cgpa` | numeric block |
| **Fairness / bias audit** | Does the model use `gender`? | `gender` vs `placed` |
| **Causal intuition & leakage** | Why can't you use `package_lpa` to predict `placed`? | see dictionary |

### Baselines (from `examples/quickstart.py`, seed 42, 75/25 split)

| Task | Model | Score |
|---|---|---|
| CGPA regression | Linear / Random Forest | R² 0.857 / 0.842 |
| Placed? (final-years, n≈2.3k) | Logistic / Random Forest | ROC-AUC 0.835 / 0.822 |
| log(package) among placed | Random Forest | R² 0.487 |
| Burnout level (3-class) | Random Forest | macro-F1 0.665 |
| Recover 7 personas | K-Means (k=7) | ARI 0.172 *(hard on purpose — beat it!)* |
| Mess review → rating | TF-IDF + Ridge | R² 0.752 |
| Bunk excuse → persona | TF-IDF + LogReg | accuracy 0.73 |

## 🏆 Challenges

Seven tasks with a fixed split, an evaluator and a leaderboard — see [`challenges/CHALLENGES.md`](challenges/CHALLENGES.md). Can you beat K-Means at finding the 7 personas (ARI 0.173)? Can you predict placement better than ROC-AUC 0.846?

## Quick start

```python
import pandas as pd
students = pd.read_csv("data/students.csv")
sems     = pd.read_csv("data/semester_records.csv")
final    = students[students.year_of_study == 4]      # placement columns exist only here
```

🤗 [Hugging Face](https://huggingface.co/datasets/pranav-singh-developer-1/desi-cs-student-survival) · `load_dataset("pranav-singh-developer-1/desi-cs-student-survival", "students")`

📊 [Kaggle](https://www.kaggle.com/datasets/pranavsinghdeveloper/desi-cs-student-survival)

```bash
pip install pandas numpy scikit-learn
python src/generate_dataset.py     # regenerate from scratch
python examples/quickstart.py      # run the baselines
```

## Sample rows (trimmed)

| student_id | residence | mess_rating | mess_review_text | cgpa | leetcode_solved | placed |
|---|---|---|---|---|---|---|
| S00009 | Hostel | 2 | "Roti could be used as a frisbee. Tbh just vibes." | 5.24 | 0 | No |
| S00010 | PG / Rented flat | 5 | "Okay food, but the queue is longer than the placement waiting list. Tbh just vibes." | 7.38 | 89 | Yes |
| S00037 | Day scholar | — | — | 5.75 | 119 | No |
