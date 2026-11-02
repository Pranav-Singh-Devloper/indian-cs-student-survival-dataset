# Data Dictionary

All data is **synthetic**. `students.csv` has 10,000 rows × 59 columns; `semester_records.csv` has 48,876 rows × 5 columns.

## `data/students.csv`

### Identity & background
| Column | Type | Description |
|---|---|---|
| student_id | str | `S00001`… (join key to `semester_records.csv`) |
| age | int | 18–24 |
| gender | cat | Male / Female / Other. **Has no causal effect on any outcome by design.** |
| home_region | cat | North, South, East, West, Central, North-East |
| home_town_type | cat | Metro, Tier-2 city, Small town, Village |
| branch | cat | CSE, IT, AI&ML, Data Science, Cyber Security, "ECE (but codes)" |
| college_tier | int | 1 (IIT/NIT-like) · 2 · 3 (the "tier-3 college, tier-1 dreams" club) |
| year_of_study | int | 1–4 |
| residence | cat | Hostel, PG / Rented flat, Day scholar |
| relationship_status | cat | Single, Situationship, Committed |

### Academics
| Column | Type | Description |
|---|---|---|
| tenth_pct, twelfth_pct | float | Board percentages (≈1.5–2% missing) |
| entrance_percentile | float | JEE/state-CET-style percentile |
| attendance_pct | float | Current overall attendance |
| bunked_classes_per_month | int | Classes skipped |
| proxy_attendance_count | int | Times someone said "yes sir" on behalf of a friend |
| study_hours_daily | float | Self-study hours/day (3% missing) |
| studies_only_night_before_exam | bool | The classic |
| all_nighters_before_exam | int | Per exam season |
| **cgpa** | float | Mean of the student's SGPAs. **Missing for ~8% (low-CGPA students are likelier to "forget" — MAR)** |
| backlogs | int | Total backlogs (KTs) so far |

### Lifestyle
| Column | Type | Description |
|---|---|---|
| sleep_hours | float | Avg per night (3% missing) |
| gaming_hours_weekly, ott_hours_weekly | float | Hours/week |
| phone_screen_time_hrs | float | Hours/day |
| chai_cups_per_day, energy_drinks_per_week, maggi_per_week | int | Survival fuel |
| delivery_orders_per_month | int | Zomato/Swiggy orders |
| monthly_pocket_money_inr | int | Rupees (5% missing) |
| wifi_speed_mbps | float | Hostel wifi is slower, obviously (4% missing) |

### Mess *(NaN for Day scholars by design — ~30%)*
| Column | Type | Description |
|---|---|---|
| mess_rating | float | 1–10 |
| food_poisoning_incidents | int | Lifetime |
| sunday_biryani_attendance_pct | float | % of Sundays attended |
| mess_review_text | str | Free-text review; sentiment tracks `mess_rating` |

### Skills & resume
| Column | Type | Description |
|---|---|---|
| leetcode_solved | int | Heavy right-skew (2% missing) |
| github_commits_last_year | int | Includes ~1% extreme outliers (2% missing) |
| projects_built, hackathons_attended, certifications_count, internships, club_memberships | int | Counts |
| linkedin_connections | int | Log-normal |
| communication_skill | float | 1–10 self-rating |
| genai_prompts_per_day | int | AI-assistant prompts a day |
| stackoverflow_copy_paste_index | float | 1–10 |
| uses_linux_btw, owns_mechanical_keyboard | bool | |
| parent_placement_queries_per_week | int | "Beta, placement ka kya hua?" |
| reason_for_last_bunk | str | Free text, relates to `archetype` |

### Mind
| Column | Type | Description |
|---|---|---|
| imposter_syndrome | float | 1–10 |
| stress_level | float | 1–10 |
| burnout_score | float | 0–100 |
| burnout_level | cat | Low (≤40) / Medium / High (>65) — derived from `burnout_score` |

### Placement *(final-year students only: `year_of_study == 4`, ~2,300 rows; NaN otherwise)*
| Column | Type | Description |
|---|---|---|
| placed | cat | Yes / No |
| placement_category | cat | Not Placed / Service Mass Recruiter / Product / Startup / Dream Product / FAANG-ish |
| offers_received | int | |
| package_lpa | float | Lakhs/yr; NaN if not placed. Right-skewed |

### Ground truth
| Column | Type | Description |
|---|---|---|
| archetype | cat | Hidden generating persona (7 classes): library_topper, backbench_legend, leetcode_grinder, hostel_socialite, certificate_collector, gaming_ghost, club_netaji. Use as the answer key for clustering, or as a multiclass target. |

## `data/semester_records.csv` (long / panel format)
| Column | Description |
|---|---|
| student_id | FK to `students.csv` |
| semester | 1–8 (a student in year *y* has `2y` records) |
| sgpa | 3.5–10 |
| backlogs_in_sem | |
| attendance_pct | Semester attendance |

`students.cgpa` is the mean of that student's `sgpa` (before the 8% of CGPAs were masked).
The generator bakes in a **semester 3–4 slump** (DSA/OS/DBMS) and **senior-year senioritis** — good for time-series/panel exercises.

## Leakage warnings
* Predicting `placed`/`placement_category`: drop `package_lpa`, `offers_received`.
* Predicting `burnout_level`: drop `burnout_score`.
* Predicting `cgpa`: `backlogs` is partly a consequence of it.
