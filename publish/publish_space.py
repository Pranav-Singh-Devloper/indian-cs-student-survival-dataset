"""Deploy the demo as a FREE static Hugging Face Space using stlite (Streamlit running in the browser).
Usage: python publish/publish_space.py <hf-user>/<space-name>"""
import shutil, sys
from pathlib import Path
import pandas as pd
from huggingface_hub import HfApi

repo_id = sys.argv[1]; root = Path(__file__).resolve().parent.parent; st = root / "hf_space_stage"
shutil.rmtree(st, ignore_errors=True); st.mkdir()
shutil.copy(root / "app" / "app.py", st / "app.py")
d = pd.read_csv(root / "data" / "students.csv"); d = d[d.year_of_study == 4]
cols = ["year_of_study", "placed", "cgpa", "leetcode_solved", "internships", "backlogs", "projects_built",
        "communication_skill", "hackathons_attended", "college_tier"]
d[cols].to_csv(st / "students_final.csv", index=False)   # small file: only what the demo needs
(st / "index.html").write_text("""<!doctype html>
<html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Will I get placed?</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.55.0/build/stlite.css"/></head>
<body><div id="root"></div>
<script src="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.55.0/build/stlite.js"></script>
<script>
async function go(){
  const [app, csv] = await Promise.all([fetch("app.py").then(r=>r.text()), fetch("students_final.csv").then(r=>r.text())]);
  stlite.mount({requirements:["pandas","numpy","scikit-learn"], entrypoint:"app.py",
                files:{"app.py":app, "students_final.csv":csv}}, document.getElementById("root"));
}
go();
</script></body></html>
""")
(st / "README.md").write_text("""---
title: Will I Get Placed?
emoji: 🎒
colorFrom: yellow
colorTo: indigo
sdk: static
pinned: false
license: cc0-1.0
---
A just-for-fun demo on the 100% synthetic Desi CS Student Survival Dataset. Runs entirely in your browser.
""")
api = HfApi(); api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(folder_path=str(st), repo_id=repo_id, repo_type="space", commit_message="Static stlite demo")
shutil.rmtree(st); print(f"https://huggingface.co/spaces/{repo_id}")
