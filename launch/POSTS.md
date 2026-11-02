# Launch posts (copy, paste, post from your own accounts)

Links used:
- GitHub: https://github.com/Pranav-Singh-Devloper/indian-cs-student-survival-dataset
- Kaggle: https://www.kaggle.com/datasets/pranavsinghdeveloper/desi-cs-student-survival
- Hugging Face: https://huggingface.co/datasets/pranav-singh-developer-1/indian-cs-student-survival
- Demo: https://huggingface.co/spaces/pranav-singh-developer-1/will-i-get-placed

Tips: post Tue–Thu, stay online for the first 2 hours to reply to every comment, and write each post differently (no copy-paste across communities).

---

## 1. LinkedIn (main launch post)

I built a dataset of 10,000 Indian CS students. None of them exist. 🎒

Every ML tutorial uses Iris, Titanic or housing prices. So I made one for us:

🍛 Mess ratings with free-text reviews ("Dal is just yellow water with trust issues")
☕ Chai cups, Maggi per week, energy drinks, all-nighters
📉 The semester 3–4 slump (DSA + OS + DBMS arrive, grades leave)
💻 LeetCode guilt, internships, backlogs, CGPA anxiety
📞 "Beta, placement ka kya hua?" — parent queries per week
🎯 Final-year placements and packages

59 features, 8 semesters of history, and 7 hidden personas (the Backbench Legend, the LeetCode Grinder, the Hostel Socialite, the Certificate Collector...) that you can try to rediscover with clustering.

It covers almost the whole classical ML syllabus: regression, classification, imbalance, clustering, PCA, NLP, time series, missing data, even a fairness audit.

I also built a tiny demo, "Will I get placed?", that runs in your browser. (Please don't take it seriously. It's synthetic. Go solve one more problem.)

Free, CC0, fully reproducible:
👉 Kaggle: [link]
👉 Hugging Face: [link]
👉 Code + challenges: [link]

Can you beat my baseline at finding the 7 personas? (ARI 0.173, and it's harder than it sounds.)

#MachineLearning #DataScience #Kaggle #Students #India #OpenData

---

## 2. X / Twitter thread

1/ I made a dataset of 10,000 Indian CS students.
Hostel. Mess. Bunking. Backlogs. LeetCode guilt. Placement season.
100% synthetic, 100% relatable. Free for ML practice 🧵

2/ Mess reviews included:
"Dal is just yellow water with trust issues."
"Roti could be used as a frisbee."
"Paneer actually had paneer. 10/10 would eat again."
(Yes, you can train a sentiment model on them.)

3/ Real patterns baked in:
• Sem 3–4 slump (DSA/OS/DBMS) 📉
• Hostel wifi < PG wifi 📶
• Placement depends on CGPA, LeetCode, internships, backlogs, tier
• Gender has ZERO causal effect, so audit your model for bias

4/ 7 hidden personas: Backbench Legend, LeetCode Grinder, Hostel Socialite, Library Topper, Gaming Ghost, Certificate Collector, Club Netaji.
Task: rediscover them with clustering. My K-Means gets ARI 0.17. Beat it.

5/ Also: a "Will I get placed?" demo that runs in your browser (please don't take it seriously) 😅
[demo link]

6/ Kaggle: [link]
HF: [link]
GitHub (generator, challenges, notebooks): [link]
CC0 licensed. RT if you've ever eaten Maggi for dinner 3 days in a row 🍜

---

## 3. Reddit r/datasets

**Title:** [Synthetic] Dataset: 10,000 Indian CS students, 59 features (hostel, mess reviews, backlogs, CGPA, placements) for practising ML

I made this dataset and wanted to share it. It's fully synthetic (disclosed in the README), CC0, and the generator script with a fixed seed is included.

- `students.csv`: 10,000 rows × 59 columns. Habits, academics, mess ratings plus free-text reviews, skills, burnout, and placement outcome/package for final-year students.
- `semester_records.csv`: 48,876 rows of semester-wise SGPA, backlogs and attendance (panel data).
- 7 hidden personas as ground truth for clustering. Built-in missingness (MCAR, MAR, structural), skewed targets, outliers and class imbalance.
- Fixed train/test split and 7 benchmark tasks with baselines (e.g. placement ROC-AUC 0.846, CGPA R² 0.84).

Not meant for conclusions about real students. It's for learning and teaching. Feedback welcome, especially on what columns or tasks you'd want added.

Kaggle: [link] · Hugging Face: [link] · GitHub: [link]

---

## 4. Reddit r/developersIndia (or r/learnmachinelearning)

**Title:** I made a free, funny ML practice dataset about Indian CS college life (hostel mess, backlogs, LeetCode, placements)

Got tired of Titanic and Iris, so I made a synthetic dataset that every Indian engineering student will find painfully relatable: mess food reviews, chai/Maggi counts, bunking excuses ("Proxy failed, rare L"), the sem 3–4 slump, parents asking about placement, and actual placement outcomes for final years.

It's good for practising regression, classification, clustering, NLP and EDA on data that doesn't put you to sleep. Disclosure: I built it and it's 100% synthetic, so it's for practice, not for predicting your real placement :)

There's also a browser demo, "Will I get placed?", and challenge tasks with baselines if you want to compete.

Links: [Kaggle] [HF] [GitHub]

Would love to hear which relatable column I missed.

---

## 5. Show HN

**Title:** Show HN: A synthetic dataset of Indian CS student life for practising ML (CC0)

I built a 10,000-row synthetic dataset of Indian CS students (hostel, mess reviews, backlogs, CGPA, LeetCode, placements) as a more engaging alternative to Iris/Titanic for teaching and practising classical ML. Everything is generated from an explicit causal model (persona → traits → habits → semester grades → skills → placement) with injected missingness and noise, and the seeded generator is included, so you can see exactly what structure is in the data.

It includes text columns, semester-level panel data, hidden cluster labels, a fixed test split, 7 benchmark tasks with baselines, and a small in-browser demo. It's not real data and shouldn't be used to draw conclusions about real people.

I'd especially like feedback on the generative assumptions and on which tasks are too easy or too hard.

GitHub: [link] · Kaggle: [link] · HF: [link]

---

## 6. Short message for professors / ML instructors (email, WhatsApp, Discord)

Hi! I made a free, CC0 synthetic dataset on Indian CS student life for ML teaching: 10,000 students, 59 features, text columns, semester-wise panel data, hidden personas for clustering, and a fixed train/test split with benchmark tasks. It's engaging for students and covers regression, classification, clustering, PCA, NLP and a fairness exercise. Feel free to use it in labs or assignments: [GitHub link]
