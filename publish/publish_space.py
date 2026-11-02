"""Deploy the Streamlit demo as a Hugging Face Space.  Usage: python publish/publish_space.py <hf-user>/<space-name>"""
import shutil, sys
from pathlib import Path
from huggingface_hub import HfApi
repo_id = sys.argv[1]; root = Path(__file__).resolve().parent.parent; st = root / "hf_space_stage"
shutil.rmtree(st, ignore_errors=True); st.mkdir()
for f in ("app.py", "requirements.txt"): shutil.copy(root / "app" / f, st / f)
shutil.copy(root / "data" / "students.csv", st / "students.csv")
(st / "README.md").write_text("""---
title: Will I Get Placed?
emoji: 🎒
colorFrom: orange
colorTo: indigo
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
license: cc0-1.0
---
A just-for-fun demo on the synthetic Desi CS Student Survival Dataset.
""")
api = HfApi(); api.create_repo(repo_id, repo_type="space", space_sdk="streamlit", exist_ok=True)
api.upload_folder(folder_path=str(st), repo_id=repo_id, repo_type="space", commit_message="Add Streamlit demo")
shutil.rmtree(st); print(f"https://huggingface.co/spaces/{repo_id}")
