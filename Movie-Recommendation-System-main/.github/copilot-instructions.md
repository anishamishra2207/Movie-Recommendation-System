# Movie Recommendation System

- Python runtime: 3.12; Spark runtime: PySpark 3.5.1 and Java 17.
- Keep recommendation and data preparation code in `src/movie_recs/`.
- Keep the distributed bipartite graph analysis in `graphx/` using Scala GraphX.
- Run Python tests with `python -m unittest discover -s tests`.
- Use `python -m movie_recs.cli` from `src/` or set `PYTHONPATH=src`.
- Do not commit generated data or Spark output directories.
