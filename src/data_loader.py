import csv
from pathlib import Path
from typing import List, Optional, Sequence


def load_workload_trace(path: Path, split: Optional[str] = "train") -> List[float]:
    values: List[float] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is not None and row["split"] != split:
                continue
            values.append(float(row["workload_norm"]))
    return values


def build_episodes(
    series: Sequence[float],
    episode_length: int,
    stride: Optional[int] = None,
    drop_last: bool = True,
) -> List[List[float]]:
    if episode_length <= 0:
        raise ValueError("episode_length must be positive")
    if stride is None:
        stride = episode_length
    if stride <= 0:
        raise ValueError("stride must be positive")

    episodes: List[List[float]] = []
    start = 0
    while start < len(series):
        end = start + episode_length
        if end > len(series):
            if drop_last:
                break
            episode = list(series[start:])
            if episode:
                episodes.append(episode)
            break
        episodes.append(list(series[start:end]))
        start += stride
    return episodes

