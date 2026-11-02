"""
Generator for the Indian CS Student Survival Dataset.

Everything here is SYNTHETIC. No real student is in this data.
The data is produced from a small causal story:

    archetype -> latent traits (aptitude, discipline, coding passion, social, chaos)
              -> habits (sleep, study, gaming, chai, maggi ...)
              -> semester SGPAs -> CGPA, backlogs
              -> skills (LeetCode, internships, projects ...)
              -> placement outcome (final year only) -> package

Run:  python src/generate_dataset.py            (writes ./data/*.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_STUDENTS = 10_000
OUT = Path(__file__).resolve().parent.parent / "data"

rng = np.random.default_rng(SEED)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clip_round(x, lo, hi, nd=0):
    return np.round(np.clip(x, lo, hi), nd)


# --------------------------------------------------------------------------
# 1. Archetypes: ground-truth clusters (hidden label for clustering practice)
#    latent means: (aptitude, discipline, coding, social, chaos)
# --------------------------------------------------------------------------
ARCHETYPES = {
    "library_topper":       (0.9,  1.2, 0.2, -0.5, -0.9),
    "backbench_legend":     (0.0, -1.0, -0.3, 0.4,  0.9),
    "leetcode_grinder":     (0.7,  0.6, 1.5, -0.4, -0.2),
    "hostel_socialite":     (-0.1, -0.3, -0.4, 1.4,  0.6),
    "certificate_collector": (0.1,  0.8, 0.1, 0.3, -0.3),
    "gaming_ghost":         (0.3, -0.7, 0.3, -0.9,  1.3),
    "club_netaji":          (0.2,  0.1, 0.0, 1.5, -0.1),
}
ARCH_NAMES = list(ARCHETYPES)
ARCH_P = [0.12, 0.20, 0.11, 0.15, 0.13, 0.14, 0.15]

n = N_STUDENTS
arch = rng.choice(ARCH_NAMES, size=n, p=ARCH_P)
means = np.array([ARCHETYPES[a] for a in arch])
lat = means + rng.normal(0, 0.75, size=(n, 5))
apt, disc, cod, soc, chaos = lat.T

# --------------------------------------------------------------------------
# 2. Demographics & background
# --------------------------------------------------------------------------
college_tier = rng.choice([1, 2, 3], size=n, p=[0.15, 0.35, 0.50])
apt = apt + np.where(college_tier == 1, 0.35, np.where(college_tier == 2, 0.0, -0.2))

year = rng.choice([1, 2, 3, 4], size=n, p=[0.27, 0.26, 0.24, 0.23])
age = clip_round(17.5 + year + rng.choice([0, 1, 2], size=n, p=[0.7, 0.25, 0.05]), 17, 25)
gender = rng.choice(["Male", "Female", "Other"], size=n, p=[0.66, 0.33, 0.01])
branch = rng.choice(["CSE", "IT", "AI&ML", "Data Science", "Cyber Security", "ECE (but codes)"],
                    size=n, p=[0.38, 0.17, 0.2, 0.11, 0.05, 0.09])
home_region = rng.choice(["North", "South", "East", "West", "Central", "North-East"],
                         size=n, p=[0.27, 0.22, 0.14, 0.2, 0.13, 0.04])
home_town_type = rng.choice(["Metro", "Tier-2 city", "Small town", "Village"],
                            size=n, p=[0.2, 0.33, 0.35, 0.12])
residence = rng.choice(["Hostel", "PG / Rented flat", "Day scholar"], size=n, p=[0.52, 0.18, 0.30])
is_hostel = residence == "Hostel"
has_mess = residence != "Day scholar"

tenth_pct = clip_round(80 + 5.5 * apt + rng.normal(0, 4.5, n), 50, 99.6, 1)
twelfth_pct = clip_round(78 + 5.5 * apt + 0.4 * (tenth_pct - 80) + rng.normal(0, 5, n), 50, 99.5, 1)
entrance_percentile = clip_round(
    72 + 11 * apt + np.where(college_tier == 1, 12, np.where(college_tier == 2, 2, -8))
    + rng.normal(0, 7, n), 20, 99.99, 2)

# --------------------------------------------------------------------------
# 3. Daily-life habits
# --------------------------------------------------------------------------
relationship_status = rng.choice(["Single", "Situationship", "Committed"], size=n, p=[0.58, 0.17, 0.25])
rel_code = pd.Series(relationship_status).map({"Single": 0, "Situationship": 0.4, "Committed": 0.3}).values

sleep_hours = clip_round(6.7 - 0.55 * chaos - 0.15 * rel_code + rng.normal(0, 0.9, n), 3, 10, 1)
study_hours_daily = clip_round(2.0 + 0.95 * disc + 0.3 * apt - 0.25 * chaos + rng.normal(0, 0.8, n), 0, 10, 1)
attendance_pct = clip_round(73 + 8 * disc - 4 * chaos - 1.5 * (year - 1) + rng.normal(0, 7, n), 30, 100, 1)
bunked_classes_per_month = np.clip(np.round((100 - attendance_pct) * 0.45 + rng.normal(0, 2, n)), 0, 40).astype(int)
proxy_attendance_count = rng.poisson(np.clip(0.2 + 1.2 * np.maximum(chaos, 0) + 0.5 * np.maximum(-disc, 0), 0.05, 6) * (1 + is_hostel))
gaming_hours_weekly = clip_round(np.maximum(0, 7 + 6 * chaos - 2.5 * disc + rng.gamma(2, 1.5, n) - 3), 0, 60, 1)
ott_hours_weekly = clip_round(np.maximum(0, 6 + 2.5 * chaos + rng.normal(0, 3, n)), 0, 40, 1)
phone_screen_time_hrs = clip_round(5.2 + 0.7 * chaos + 0.5 * soc + rng.normal(0, 1.3, n), 1, 14, 1)
chai_cups_per_day = rng.poisson(np.clip(2.0 + 0.6 * chaos + 0.4 * is_hostel + 0.3 * (1 - sleep_hours / 7), 0.2, 9))
energy_drinks_per_week = rng.poisson(np.clip(1.0 + 1.1 * np.maximum(chaos, 0) + 0.5 * (year >= 3), 0.1, 10))
maggi_per_week = rng.poisson(np.clip(1.5 + 2.2 * is_hostel + 0.8 * (residence == "PG / Rented flat") + 0.5 * chaos, 0.2, 14))
delivery_orders_per_month = rng.poisson(np.clip(3 + 4 * (college_tier == 1) + 3 * soc + 2 * (residence == "PG / Rented flat") + 2 * is_hostel, 0, 40))
all_nighters_before_exam = np.clip(np.round(1.5 + 1.0 * chaos - 0.8 * disc + rng.normal(0, 1, n)), 0, 6).astype(int)
studies_only_night_before = (sigmoid(-0.2 - 1.3 * disc + 0.6 * chaos + rng.normal(0, 0.8, n)) > rng.random(n))
monthly_pocket_money_inr = (np.round(
    np.exp(8.55 + 0.35 * (college_tier == 1) + 0.25 * (home_town_type == "Metro") - 0.15 * (home_town_type == "Village")
           + rng.normal(0, 0.35, n)) / 100) * 100).astype(int)
wifi_speed_mbps = clip_round(np.where(is_hostel, rng.lognormal(2.7, 0.8, n), rng.lognormal(3.4, 0.7, n)), 0.5, 300, 1)

# Mess (Hostel & PG only)
mess_rating = clip_round(5.2 - 0.9 * (college_tier == 3) + 0.6 * (college_tier == 1)
                         + 0.5 * (home_region == "South") * 0 + rng.normal(0, 1.9, n), 1, 10, 0)
mess_rating = np.where(has_mess, mess_rating, np.nan)
food_poisoning_incidents = np.where(
    has_mess, rng.poisson(np.clip(0.15 + (10 - np.nan_to_num(mess_rating, nan=5)) * 0.08, 0.02, 2.0) * (1 + 0.3 * (year - 1))), np.nan)
sunday_biryani_attendance = np.where(has_mess, np.clip(np.round(60 + 25 * (mess_rating > 5) + rng.normal(0, 15, n)), 0, 100), np.nan)
sunday_biryani_attendance = np.where(has_mess, np.clip(np.nan_to_num(sunday_biryani_attendance), 10, 100), np.nan)

# --------------------------------------------------------------------------
# 4. Semester records (long format) -> CGPA & backlogs
# --------------------------------------------------------------------------
base_gpa = (7.2 + 0.50 * apt + 0.30 * disc - 0.15 * chaos + 0.006 * (attendance_pct - 73)
            + 0.02 * (study_hours_daily - 2.5) + 0.1 * (college_tier == 1) - 0.05 * (gaming_hours_weekly > 25)
            + 0.03 * (sleep_hours - 6.5))
trend = 0.10 * disc - 0.10 * chaos  # per-year drift: grinders improve, chaos decays
rows = []
cgpa = np.full(n, np.nan)
total_backlogs = np.zeros(n, dtype=int)
n_sems = 2 * year  # assume the student has just finished the year
for i in range(n):
    sgpas = []
    for s in range(1, n_sems[i] + 1):
        slump = -0.30 if s in (3, 4) else 0.0       # the DSA / OS / DBMS reality check
        lazy_end = -0.15 if s in (7, 8) else 0.0    # senioritis
        mu = base_gpa[i] + trend[i] * (s / 2 - 1) + slump + lazy_end
        sg = float(np.clip(mu + rng.normal(0, 0.38), 3.5, 10))
        bl_rate = np.exp(-2.2 + 0.6 * (6.6 - sg) + 0.25 * chaos[i] - 0.3 * disc[i] + (0.35 if s in (3, 4) else 0))
        bl = int(min(rng.poisson(bl_rate), 5))
        att = float(np.clip(attendance_pct[i] + rng.normal(0, 5) - 0.6 * s, 30, 100))
        sg = max(3.5, sg - 0.12 * bl)
        sgpas.append(sg)
        rows.append((i, s, round(sg, 2), bl, round(att, 1)))
    cgpa[i] = np.mean(sgpas)
    total_backlogs[i] = sum(r[3] for r in rows[-n_sems[i]:])
cgpa = np.round(cgpa, 2)

sem_df = pd.DataFrame(rows, columns=["student_idx", "semester", "sgpa", "backlogs_in_sem", "attendance_pct"])
student_ids = np.array([f"S{i+1:05d}" for i in range(n)])
sem_df.insert(0, "student_id", student_ids[sem_df.pop("student_idx").values])

# --------------------------------------------------------------------------
# 5. Skills & resume
# --------------------------------------------------------------------------
yr_f = (year - 1) / 3.0
leetcode_solved = np.round(np.maximum(0, np.exp(2.2 + 1.05 * cod + 0.5 * yr_f * 2 + 0.2 * disc + rng.normal(0, 0.8, n)) - 5)).astype(int)
leetcode_solved = np.minimum(leetcode_solved, 1800)
github_commits_last_year = np.round(np.exp(3.0 + 0.9 * cod + 0.3 * yr_f * 2 + rng.normal(0, 1.0, n))).astype(int) - 1
github_commits_last_year = np.clip(github_commits_last_year, 0, 3000)
projects_built = rng.poisson(np.clip(0.4 + 0.7 * (year - 1) * 0.6 + 0.6 * cod + 0.2 * disc, 0.05, 9))
hackathons_attended = rng.poisson(np.clip(0.3 + 0.35 * (year - 1) + 0.4 * soc + 0.4 * cod, 0.02, 8))
certifications_count = rng.poisson(np.clip(1.0 + 0.5 * (year - 1) + 1.3 * disc + (arch == "certificate_collector") * 3.5, 0.1, 25))
internships = rng.poisson(np.clip(np.where(year >= 3, 0.15 + 0.35 * (year - 3) + 0.3 * cod + 0.2 * apt
                                            + 0.2 * (college_tier == 1) + 0.05 * soc, 0.01), 0.005, 4))
club_memberships = rng.poisson(np.clip(0.7 + 0.8 * soc + 0.1 * disc, 0.05, 6))
linkedin_connections = np.clip(np.round(np.exp(4.4 + 0.55 * soc + 0.35 * yr_f * 2 + 0.15 * cod + rng.normal(0, 0.7, n))), 5, 30000).astype(int)
communication_skill = clip_round(5 + 1.1 * soc + 0.5 * apt + 0.2 * yr_f * 2 + rng.normal(0, 1.4, n), 1, 10)
genai_prompts_per_day = rng.poisson(np.clip(5 + 3 * (year - 1) * 0.5 + 2 * cod + 3 * chaos * 0.6, 0.3, 80))
stackoverflow_copy_paste_index = clip_round(5 - 0.8 * apt - 0.4 * cod + 0.5 * chaos + rng.normal(0, 1.4, n), 1, 10)
uses_linux_btw = sigmoid(-1.2 + 0.9 * cod + 0.3 * apt + rng.normal(0, 0.6, n)) > rng.random(n)
owns_mechanical_keyboard = sigmoid(-1.8 + 0.7 * cod + 0.5 * chaos + rng.normal(0, 0.7, n)) > rng.random(n)
parent_placement_queries_per_week = rng.poisson(np.clip(0.5 + 1.0 * (year - 1) + 0.5 * (year == 4) * 2 + 0.6 * (college_tier == 3), 0.2, 15))

# --------------------------------------------------------------------------
# 6. Mental health: stress, imposter syndrome, burnout
# --------------------------------------------------------------------------
imposter_syndrome = clip_round(4.5 + 0.55 * apt * 0 + 0.7 * (1 - cod) * 0.5 + 0.35 * (year - 2) - 0.3 * (communication_skill - 5) * 0.4
                               + 0.6 * (college_tier == 1) + rng.normal(0, 1.5, n), 1, 10)
stress_level = clip_round(4.0 + 0.45 * chaos + 0.35 * (year - 2.5) - 0.25 * (sleep_hours - 6.5) + 0.35 * np.maximum(total_backlogs, 0) ** 0.5
                          + 0.12 * parent_placement_queries_per_week + 0.25 * (study_hours_daily > 6) * 1
                          + 0.35 * (attendance_pct < 60) + rng.normal(0, 1.3, n), 1, 10)
burnout_score = clip_round(18 + 5.5 * stress_level - 3.2 * (sleep_hours - 6) + 1.7 * imposter_syndrome
                           + 0.8 * energy_drinks_per_week + 0.15 * phone_screen_time_hrs * 2 - 0.8 * (club_memberships > 0)
                           - 2.0 * rel_code * (relationship_status == "Committed") + rng.normal(0, 7, n), 0, 100, 1)
burnout_level = pd.cut(burnout_score, [-1, 40, 65, 101], labels=["Low", "Medium", "High"]).astype(str)

# --------------------------------------------------------------------------
# 7. Placement (final-year students only) -> classification / regression
# --------------------------------------------------------------------------
final = year == 4
score = (1.1 * (cgpa - 7.2)
         + 0.55 * np.log1p(leetcode_solved) - 2.0
         + 0.55 * internships
         + 0.25 * projects_built ** 0.5
         + 0.12 * (communication_skill - 5)
         + 0.35 * (college_tier == 1) - 0.55 * (college_tier == 3)
         - 0.55 * np.minimum(total_backlogs, 4)
         + 0.1 * hackathons_attended ** 0.5
         + 0.35 * (certifications_count > 5) * 0
         + rng.normal(0, 0.9, n))
placed = sigmoid(0.9 * score + 1.1) > rng.random(n)

# which company bucket
prod_score = score + 0.9 * np.log1p(leetcode_solved) / 3 + 0.25 * (college_tier == 1)
tier_cut_hi, tier_cut_mid = np.quantile(prod_score[placed & final], [0.90, 0.68])
placement_category = np.where(~placed, "Not Placed",
                      np.where(prod_score > tier_cut_hi, "Dream Product / FAANG-ish",
                      np.where(prod_score > tier_cut_mid, "Product / Startup", "Service Mass Recruiter")))
base_pkg = {"Service Mass Recruiter": 3.8, "Product / Startup": 10.0, "Dream Product / FAANG-ish": 26.0}
package_lpa = np.full(n, np.nan)
for cat, mu in base_pkg.items():
    m = placement_category == cat
    package_lpa[m] = mu * np.exp(rng.normal(0, 0.22 if cat == "Service Mass Recruiter" else 0.38, m.sum())
                                 + 0.05 * (college_tier[m] == 1))
package_lpa = np.round(np.clip(package_lpa, 2.4, 120), 2)
offers_received = np.where(placed, 1 + rng.poisson(np.where(placement_category == "Service Mass Recruiter", 0.4, 0.7)), 0)
offers_received = np.where(placement_category == "Not Placed", rng.poisson(0.05, n), offers_received)  # off-campus rejections lol
placed_str = np.where(placed, "Yes", "No").astype(object)

# non-final-year students: placement columns are NaN by design
for arr_name in ("placed_str", "placement_category", "package_lpa", "offers_received"):
    pass
placed_out = np.where(final, placed_str, None)
placement_category_out = np.where(final, placement_category, None)
package_out = np.where(final, package_lpa, np.nan)
offers_out = np.where(final, offers_received, np.nan)

# --------------------------------------------------------------------------
# 8. Text columns (NLP practice)
# --------------------------------------------------------------------------
REVIEW_LOW = [
    "Dal is just yellow water with trust issues.", "Found something in the sabzi that was not on the menu.",
    "Roti could be used as a frisbee.", "Same aloo gobhi for the 9th day. I can taste regret.",
    "Mess uncle says 'paneer' like it is a rumour.", "Hygiene level: Chernobyl.",
    "Rice had more stones than my college has placements.", "Chai was lukewarm sadness.",
]
REVIEW_MID = [
    "Edible. Sunday biryani saves the week.", "Not great, not terrible. Survival mode on.",
    "Dal is fine, sabzi is a mystery, but the pickle slaps.", "Average mess. Maggi from the room is the real MVP.",
    "Okay food, but the queue is longer than the placement waiting list.", "Decent breakfast, tragic dinner.",
    "It is food. That is all I can say.",
]
REVIEW_HIGH = [
    "Paneer actually had paneer. 10/10 would eat again.", "Mess uncle is a legend, extra roti without asking.",
    "Surprisingly clean and tasty, mummy ke haath jaisa.", "Sunday special was chef's kiss.",
    "Best mess in the campus, fight me.", "Gulab jamun on Wednesday?! Mess committee finally woke up.",
    "Fresh, hot and not a single hair. Miracle.",
]
REVIEW_TAIL = ["", "", "", " Wifi was down too.", " Still better than Maggi.", " Hostel life, bro.",
               " Sent a complaint, got a 'seen'.", " Tbh just vibes.", " Warden did not care."]


def review(r):
    if np.isnan(r):
        return None
    pool = REVIEW_LOW if r <= 3 else REVIEW_MID if r <= 7 else REVIEW_HIGH
    return rng.choice(pool) + rng.choice(REVIEW_TAIL)


mess_review_text = np.array([review(r) for r in mess_rating], dtype=object)

EXCUSES = {
    "library_topper":       ["Was in the library reading a reference book on top of the reference book.",
                             "Went to ask doubts to the professor for the 5th time.",
                             "Skipped to finish the assignment due next month."],
    "backbench_legend":     ["Alarm did not ring (I did not set it).", "Proxy failed, rare L.",
                             "Fever. 103. Only during the 9 AM lecture.", "Slept through the bell. And the next three."],
    "leetcode_grinder":     ["Maintaining my 400-day streak, class can wait.", "Contest was at the same time as the lecture.",
                             "Was stuck on a Medium that felt Hard."],
    "hostel_socialite":     ["Roommate's birthday celebration went till 5 AM.", "Was organising the hostel night.",
                             "Chai tapri discussion turned into a TED talk."],
    "certificate_collector": ["Finishing a 4-hour Coursera course on Prompt Engineering.",
                              "Was adding the 31st certificate to LinkedIn.", "Had a webinar. Certificate pending."],
    "gaming_ghost":         ["Ranked match. Could not leave the lobby.", "Server maintenance, but for me, in real life.",
                             "Lag spiked, mood crashed, class skipped."],
    "club_netaji":          ["Poster for the fest was due.", "Meeting with the sponsor. Very important. Very vague.",
                             "Was busy convincing the dean about the event budget."],
}
GENERIC = ["Wifi down during online quiz.", "Mess ka khana was suspicious, stomach said no.",
           "Bus got late, lecture got skipped, karma balanced.", "Was at the placement cell asking about companies.",
           "Laptop hang ho gaya, project submission jaldi thi.", "Thoda headache tha. Thoda zyada nahi tha."]
reason_last_bunk = np.array(
    [rng.choice(GENERIC) if rng.random() < 0.35 else rng.choice(EXCUSES[a]) for a in arch], dtype=object)

# --------------------------------------------------------------------------
# 9. Assemble, then add realistic messiness (missing values)
# --------------------------------------------------------------------------
students = pd.DataFrame({
    "student_id": student_ids,
    "age": age.astype(int),
    "gender": gender,
    "home_region": home_region,
    "home_town_type": home_town_type,
    "branch": branch,
    "college_tier": college_tier,
    "year_of_study": year,
    "residence": residence,
    "relationship_status": relationship_status,
    "tenth_pct": tenth_pct,
    "twelfth_pct": twelfth_pct,
    "entrance_percentile": entrance_percentile,
    "attendance_pct": attendance_pct,
    "bunked_classes_per_month": bunked_classes_per_month,
    "proxy_attendance_count": proxy_attendance_count,
    "study_hours_daily": study_hours_daily,
    "studies_only_night_before_exam": studies_only_night_before,
    "all_nighters_before_exam": all_nighters_before_exam,
    "sleep_hours": sleep_hours,
    "gaming_hours_weekly": gaming_hours_weekly,
    "ott_hours_weekly": ott_hours_weekly,
    "phone_screen_time_hrs": phone_screen_time_hrs,
    "chai_cups_per_day": chai_cups_per_day,
    "energy_drinks_per_week": energy_drinks_per_week,
    "maggi_per_week": maggi_per_week,
    "delivery_orders_per_month": delivery_orders_per_month,
    "monthly_pocket_money_inr": monthly_pocket_money_inr,
    "wifi_speed_mbps": wifi_speed_mbps,
    "mess_rating": mess_rating,
    "food_poisoning_incidents": food_poisoning_incidents,
    "sunday_biryani_attendance_pct": sunday_biryani_attendance,
    "mess_review_text": mess_review_text,
    "reason_for_last_bunk": reason_last_bunk,
    "cgpa": cgpa,
    "backlogs": total_backlogs,
    "leetcode_solved": leetcode_solved,
    "github_commits_last_year": github_commits_last_year,
    "projects_built": projects_built,
    "hackathons_attended": hackathons_attended,
    "certifications_count": certifications_count,
    "internships": internships,
    "club_memberships": club_memberships,
    "linkedin_connections": linkedin_connections,
    "communication_skill": communication_skill,
    "genai_prompts_per_day": genai_prompts_per_day,
    "stackoverflow_copy_paste_index": stackoverflow_copy_paste_index,
    "uses_linux_btw": uses_linux_btw,
    "owns_mechanical_keyboard": owns_mechanical_keyboard,
    "parent_placement_queries_per_week": parent_placement_queries_per_week,
    "imposter_syndrome": imposter_syndrome,
    "stress_level": stress_level,
    "burnout_score": burnout_score,
    "burnout_level": burnout_level,
    "placed": placed_out,
    "placement_category": placement_category_out,
    "offers_received": offers_out,
    "package_lpa": package_out,
    "archetype": arch,
})

# MCAR holes: people skip survey questions
for col, p in [("sleep_hours", 0.03), ("study_hours_daily", 0.03), ("leetcode_solved", 0.02),
               ("wifi_speed_mbps", 0.04), ("tenth_pct", 0.015), ("twelfth_pct", 0.02),
               ("monthly_pocket_money_inr", 0.05), ("github_commits_last_year", 0.02)]:
    students.loc[rng.random(n) < p, col] = np.nan
# MAR hole: students with low CGPA "forget" to report it (ego protection)
low = (students["cgpa"] < 6.3) & (rng.random(n) < 0.18)
students.loc[low, "cgpa"] = np.nan
# 1% wild-but-plausible outliers (the guy who has 2000 commits "for fun")
idx = rng.choice(n, size=int(0.01 * n), replace=False)
students.loc[idx, "github_commits_last_year"] = rng.integers(1500, 6000, size=len(idx))

# int-like columns that picked up NaN -> nullable ints
for c in ["leetcode_solved", "github_commits_last_year", "monthly_pocket_money_inr", "food_poisoning_incidents", "offers_received"]:
    students[c] = students[c].astype("Int64")

OUT.mkdir(exist_ok=True)
students.to_csv(OUT / "students.csv", index=False)
sem_df.to_csv(OUT / "semester_records.csv", index=False)
print("students:", students.shape, " semester_records:", sem_df.shape)
