"""🎒 Will I get placed? — playful demo built on the Indian CS Student Survival Dataset (synthetic!)."""
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="Will I get placed?", page_icon="🎒")
DATA = next(p for p in [Path("students_final.csv"), Path(__file__).parent / "students.csv", Path(__file__).parent.parent / "data" / "students.csv"] if p.exists())

FEATURES = ["cgpa", "leetcode_solved", "internships", "backlogs", "projects_built", "communication_skill",
            "hackathons_attended", "college_tier"]

@st.cache_resource
def train():
    d = pd.read_csv(DATA); d = d[d.year_of_study == 4].dropna(subset=FEATURES + ["placed"])
    m = GradientBoostingClassifier(random_state=0).fit(d[FEATURES], (d.placed == "Yes").astype(int))
    return m, d

model, d = train()
st.title("🎒 Will I get placed?")
st.caption("A just-for-fun demo trained on a **100% synthetic** dataset. It says nothing about real life, your college, or your future. Go solve one more LeetCode anyway.")
c1, c2 = st.columns(2)
cgpa = c1.slider("CGPA", 4.0, 10.0, 7.2, 0.1)
lc = c2.slider("LeetCode problems solved", 0, 1000, 80, 10)
intern = c1.slider("Internships", 0, 4, 0)
back = c2.slider("Backlogs", 0, 8, 0)
proj = c1.slider("Projects built", 0, 10, 2)
comm = c2.slider("Communication skill (1-10)", 1, 10, 6)
hack = c1.slider("Hackathons attended", 0, 8, 1)
tier = c2.selectbox("College tier", [1, 2, 3], index=1)

x = pd.DataFrame([[cgpa, lc, intern, back, proj, comm, hack, tier]], columns=FEATURES)
p = float(model.predict_proba(x)[0, 1])
st.metric("Placement probability (synthetic world)", f"{p:.0%}")
st.progress(p)
st.write("🔥 Offer letter incoming." if p > .8 else "🤞 Decent odds. Keep grinding." if p > .5 else
         "😬 Time to fix the resume and rethink the Maggi-to-LeetCode ratio." if p > .25 else "🫠 Mess food and prayers.")
peers = (d.placed == "Yes").groupby(pd.cut(d.cgpa, [0, 6, 7, 8, 10])).mean().round(2)
st.subheader("Placement rate by CGPA band (final-years)")
st.bar_chart(peers.rename(index=str))
st.markdown("[Dataset on Hugging Face](https://huggingface.co/datasets/pranav-singh-developer-1/indian-cs-student-survival) · "
            "[Kaggle](https://www.kaggle.com/datasets/pranavsinghdeveloper/desi-cs-student-survival) · "
            "[GitHub](https://github.com/Pranav-Singh-Devloper/indian-cs-student-survival-dataset)")
