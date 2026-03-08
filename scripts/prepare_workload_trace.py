#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized workload trace from Google task_events.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--pattern", type=str, default="task_events_part_*.csv.gz")
    parser.add_argument("--max-files", type=int, default=20)
    parser.add_argument("--bucket-seconds", type=int, default=60)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--norm-quantile", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=Path("data/processed/workload_trace.csv"))
    return parser.parse_args()


def safe_float(text: str, default: float = 0.0) -> float:
    if text is None or text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def safe_int(text: str):
    if text is None or text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if args.bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be positive")
    if not 0.0 < args.norm_quantile <= 1.0:
        raise ValueError("norm_quantile must be in (0, 1]")

    files = sorted(args.raw_dir.glob(args.pattern))
    if args.max_files is not None and args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files found in {args.raw_dir} matching {args.pattern}")

    # bucket_idx -> (cpu_submit_sum, submit_count)
    buckets: DefaultDict[int, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    total_rows = 0
    submit_rows = 0

    for path in files:
        with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                total_rows += 1
                if len(row) < 10:
                    continue
                event_type = row[5].strip()
                if event_type != "0":
                    continue

                timestamp_us = safe_int(row[0])
                if timestamp_us is None:
                    continue

                cpu_request = safe_float(row[9], default=0.0)
                bucket_idx = (timestamp_us // 1_000_000) // args.bucket_seconds
                prev_cpu, prev_count = buckets[bucket_idx]
                buckets[bucket_idx] = (prev_cpu + cpu_request, prev_count + 1)
                submit_rows += 1

    if not buckets:
        raise RuntimeError("No submit events found; preprocessing produced empty buckets.")

    min_bucket = min(buckets.keys())
    max_bucket = max(buckets.keys())
    dense_rows = []
    cpu_values = []
    for bucket_idx in range(min_bucket, max_bucket + 1):
        cpu_sum, submit_count = buckets.get(bucket_idx, (0.0, 0))
        dense_rows.append(
            {
                "bucket_id": bucket_idx,
                "start_time_sec": bucket_idx * args.bucket_seconds,
                "cpu_submit_sum": cpu_sum,
                "submit_count": submit_count,
            }
        )
        cpu_values.append(cpu_sum)

    min_cpu = min(cpu_values)
    max_cpu = max(cpu_values)
    sorted_cpu = sorted(cpu_values)
    q_index = int((len(sorted_cpu) - 1) * args.norm_quantile)
    norm_scale = sorted_cpu[q_index]
    if norm_scale <= 1e-12:
        norm_scale = max_cpu if max_cpu > 1e-12 else 1.0
    for row in dense_rows:
        row["workload_norm"] = min(row["cpu_submit_sum"] / norm_scale, 1.0)

    split_idx = int(len(dense_rows) * args.train_ratio)
    split_idx = max(1, min(split_idx, len(dense_rows) - 1))
    for idx, row in enumerate(dense_rows):
        row["split"] = "train" if idx < split_idx else "test"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket_id",
        "start_time_sec",
        "cpu_submit_sum",
        "submit_count",
        "workload_norm",
        "split",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dense_rows)

    meta_path = args.output.with_suffix(".meta.json")
    metadata: Dict[str, object] = {
        "raw_dir": str(args.raw_dir),
        "file_pattern": args.pattern,
        "files_used": [str(p) for p in files],
        "bucket_seconds": args.bucket_seconds,
        "train_ratio": args.train_ratio,
        "norm_quantile": args.norm_quantile,
        "total_rows_scanned": total_rows,
        "submit_rows_used": submit_rows,
        "num_time_buckets": len(dense_rows),
        "train_points": split_idx,
        "test_points": len(dense_rows) - split_idx,
        "min_cpu_submit_sum": min_cpu,
        "max_cpu_submit_sum": max_cpu,
        "norm_scale_cpu_submit_sum": norm_scale,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {args.output} with {len(dense_rows)} points from {len(files)} files.")
    print(f"Wrote metadata to {meta_path}.")


if __name__ == "__main__":
    main()
