"""Publish to the Hugging Face Hub.  Usage: python publish/publish_hf.py <hf-username>/<dataset-name>
Needs `huggingface-cli login` with a token that has WRITE access."""
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo_id = sys.argv[1]
root = Path(__file__).resolve().parent.parent
stage = root / "hf_stage"
shutil.rmtree(stage, ignore_errors=True)
(stage / "data").mkdir(parents=True)
for f in ("students.csv", "semester_records.csv", "splits.csv"):
    shutil.copy(root / "data" / f, stage / "data" / f)
shutil.copy(root / "DATA_DICTIONARY.md", stage)
shutil.copy(root / "challenges" / "CHALLENGES.md", stage)
shutil.copy(root / "src" / "generate_dataset.py", stage)

card = """---
license: cc0-1.0
pretty_name: Desi CS Student Survival Dataset
language:
- en
size_categories:
- 10K<n<100K
task_categories:
- tabular-classification
- tabular-regression
- text-classification
- feature-extraction
tags:
- education
- india
- synthetic
- humor
- students
- beginner-friendly
configs:
- config_name: students
  data_files: data/students.csv
  default: true
- config_name: semester_records
  data_files: data/semester_records.csv
- config_name: splits
  data_files: data/splits.csv
---

"""
body = (root / "README.md").read_text()
GH = "https://github.com/Pranav-Singh-Devloper/desi-cs-student-survival-dataset/blob/main/"
body = body.replace("](src/generate_dataset.py)", "](generate_dataset.py)").replace("](challenges/CHALLENGES.md)", "](CHALLENGES.md)").replace("](data/splits.csv)", "](data/splits.csv)").replace("](examples/quickstart.py)", f"]({GH}examples/quickstart.py)")
(stage / "README.md").write_text(card + body)

api = HfApi()
api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
api.upload_folder(folder_path=str(stage), repo_id=repo_id, repo_type="dataset", commit_message="Add Desi CS Student Survival Dataset")
shutil.rmtree(stage)
print(f"https://huggingface.co/datasets/{repo_id}")
