#!/usr/bin/env bash
# Usage: KAGGLE_USER=<kaggle-username> ./publish/publish_kaggle.sh [new|version]
# Needs `pip install kaggle` and ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY env vars).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}"
STAGE=kaggle_stage; rm -rf "$STAGE"; mkdir "$STAGE"
cp data/students.csv data/semester_records.csv DATA_DICTIONARY.md src/generate_dataset.py "$STAGE"/
python3 publish/build_metadata.py "$KAGGLE_USER" "$STAGE"
if [ "${1:-new}" = "new" ]; then
  kaggle datasets create -p "$STAGE" --dir-mode zip
else
  kaggle datasets version -p "$STAGE" -m "${2:-Update}" --dir-mode zip
fi
rm -rf "$STAGE"
