"""Run repeated local/cluster parameter sweeps and write a CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/demo")
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--masters", nargs="+", default=["local[2]", "local[*]"])
    parser.add_argument("--partitions", nargs="+", type=int, default=[2, 8])
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    rows = []
    for master in args.masters:
        for partitions in args.partitions:
            for rank in args.ranks:
                for repeat in range(args.repeats):
                    run_dir = output_dir / f"{master.replace('[', '').replace(']', '').replace('*', 'all')}-p{partitions}-r{rank}-n{repeat + 1}"
                    command = [sys.executable, "-m", "movie_recs.cli", "--data-dir", args.data_dir,
                               "--output-dir", str(run_dir), "--master", master, "--rank", str(rank),
                               "--iterations", str(args.iterations), "--partitions", str(partitions)]
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
                    started = time.perf_counter()
                    subprocess.run(command, cwd=project_root, env=env, check=True, capture_output=True, text=True)
                    wall_seconds = time.perf_counter() - started
                    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                    rows.append({"master": master, "partitions": partitions, "rank": rank,
                                 "repeat": repeat + 1, "wall_seconds": wall_seconds,
                                 "fit_seconds": metrics["timing_seconds"]["model_fit"],
                                 "total_spark_seconds": metrics["timing_seconds"]["total"],
                                 "rmse": metrics["als"]["rmse"], "mae": metrics["als"]["mae"],
                                 "precision_at_k": metrics["als"]["precision_at_k"],
                                 "rating_count": metrics["dataset"]["ratings"]})
                    print(json.dumps(rows[-1]))
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
