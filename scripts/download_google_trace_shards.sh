#!/usr/bin/env bash
set -euo pipefail

START_INDEX="${1:-0}"
END_INDEX="${2:-19}"
OUT_DIR="${3:-data/raw}"

mkdir -p "${OUT_DIR}"

for idx in $(seq -f "%05g" "${START_INDEX}" "${END_INDEX}"); do
  out_path="${OUT_DIR}/task_events_part_${idx}.csv.gz"
  url="https://storage.googleapis.com/clusterdata-2011-2/task_events/part-${idx}-of-00500.csv.gz"
  if [[ -f "${out_path}" ]]; then
    echo "Skip existing ${out_path}"
    continue
  fi
  echo "Downloading ${url}"
  curl -L -o "${out_path}" "${url}"
done

echo "Done. Files in ${OUT_DIR}:"
ls -lh "${OUT_DIR}"

