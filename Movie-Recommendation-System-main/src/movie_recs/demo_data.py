"""Generate deterministic, MovieLens-shaped CSV data for a local smoke test."""

from __future__ import annotations

import csv
import random
from pathlib import Path


def generate_demo_data(output_dir: str | Path, users: int = 40, movies: int = 30) -> None:
    if users < 4 or movies < 5:
        raise ValueError("Demo data requires at least 4 users and 5 movies")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(23)
    with (destination / "movies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["movieId", "title", "genres"])
        for movie_id in range(1, movies + 1):
            writer.writerow([movie_id, f"Film {movie_id:03d}", ["Drama", "Comedy", "Sci-Fi"][movie_id % 3]])
    with (destination / "users.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["userId", "age", "occupation", "region"])
        for user_id in range(1, users + 1):
            writer.writerow([user_id, 18 + (user_id * 7) % 53, ["student", "engineer", "artist"][user_id % 3], ["north", "south", "west"][user_id % 3]])
    with (destination / "ratings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["userId", "movieId", "rating", "timestamp"])
        for user_id in range(1, users + 1):
            # Overlapping taste groups create meaningful co-rating relationships.
            group = (user_id - 1) // max(1, users // 4)
            preferred = set(range(1 + group * 4, min(movies, 4 + group * 4) + 1))
            selected = set(rng.sample(range(1, movies + 1), min(12, movies))) | preferred
            for movie_id in sorted(selected):
                base = 4.0 if movie_id in preferred else 2.5
                writer.writerow([user_id, movie_id, min(5.0, max(0.5, base + rng.choice([-1, 0, 0, 0, 1]) * 0.5)), 1_700_000_000 + user_id * 100 + movie_id])
