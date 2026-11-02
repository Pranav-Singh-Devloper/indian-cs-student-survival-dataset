"""Creates data/splits.csv: a fixed 80/20 train/test split so everyone's scores are comparable."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
root = Path(__file__).resolve().parent.parent
d = pd.read_csv(root / "data" / "students.csv", usecols=["student_id", "archetype"])
tr, te = train_test_split(d, test_size=0.2, random_state=2026, stratify=d.archetype)
out = pd.concat([tr.assign(split="train"), te.assign(split="test")])[["student_id", "split"]].sort_values("student_id")
out.to_csv(root / "data" / "splits.csv", index=False); print(out.split.value_counts().to_dict())
