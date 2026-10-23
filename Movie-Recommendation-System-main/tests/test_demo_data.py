import csv
import tempfile
import unittest
from pathlib import Path

from movie_recs.demo_data import generate_demo_data


class DemoDataTests(unittest.TestCase):
    def test_generates_joinable_movielens_shape_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            generate_demo_data(temp_dir, users=8, movies=12)
            path = Path(temp_dir)
            with (path / "ratings.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with (path / "users.csv").open(encoding="utf-8", newline="") as handle:
                users = list(csv.DictReader(handle))
            with (path / "movies.csv").open(encoding="utf-8", newline="") as handle:
                movies = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertEqual(len(users), 8)
            self.assertEqual(len(movies), 12)
            self.assertTrue({row["userId"] for row in rows} <= {row["userId"] for row in users})
            self.assertTrue({row["movieId"] for row in rows} <= {row["movieId"] for row in movies})

    def test_rejects_too_small_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                generate_demo_data(temp_dir, users=2, movies=3)


if __name__ == "__main__":
    unittest.main()
