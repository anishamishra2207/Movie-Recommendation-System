# Distributed Movie Recommendation and Behavior Analysis

A cluster-ready reference implementation using Spark SQL for ingestion and preprocessing, PySpark MLlib ALS for collaborative filtering, and Scala GraphX for distributed bipartite graph analysis. The local demo creates deterministic synthetic MovieLens-shaped CSVs; provide your own compatible CSVs for real experiments.

## Architecture

```mermaid
flowchart LR
    A[ratings.csv] --> B[Spark SQL schema validation and cleaning]
    U[users.csv] --> C[User and movie enrichment]
    M[movies.csv] --> C
    B --> D[80/20 seeded split]
    D --> E[MLlib ALS]
    E --> F[RMSE and MAE]
    E --> G[Precision@K, Recall@K, Hit Rate]
    E --> H[Per-user recommendations]
    C --> I[Popularity and behavior profiles]
    A --> J[GraphX bipartite graph]
    J --> K[Label propagation communities]
    J --> L[Movie/user degree patterns]
    F --> O[Metrics and benchmark outputs]
    G --> O
    I --> O
    K --> O
    L --> O
```

Ratings use integer user/movie keys and a rating in `[0.5, 5.0]`. GraphX allocates disjoint vertex ID ranges for users and movies and stores each rating edge in both directions, so degree counts are intuitive for both partitions. Label propagation is an unsupervised structural signal, not a guarantee of semantic preference communities; validate it against domain-specific cohorts before product use.

## Team task allocation

| Member | Responsibilities | Deliverables |
|---|---|---|
| Sahil : PySpark and recommendations | Implement CSV ingestion, validation, cleaning, user/movie behavior profiles, ALS training, rating and ranking evaluation, and benchmark experiments. | `src/movie_recs/`, `experiments/benchmark.py`, related Python tests, and recommendation/evaluation results under `outputs/`. |
| Anisha : Scala GraphX analysis | Build the user–movie bipartite graph, analyze movie/user degree patterns, run label propagation, and validate GraphX output on the shared ratings data. | `graphx/`, community/popularity outputs under `outputs/graphx/`, and GraphX run/build notes. |

Both members should agree on the CSV schemas and Spark/Scala versions, run their component tests, and review the README and final outputs together before presenting the results. Replace the generic member labels with the assigned names when the team is confirmed.

## Prerequisites

- Python 3.12
- Java 17
- PySpark 3.5.1 (`pip install -r requirements.txt`)
- For GraphX: sbt 1.x and Spark 3.5.1 available on the cluster

## Local run

From the repository root in PowerShell:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m movie_recs.cli --generate-demo --data-dir data/demo --output-dir outputs/demo
```

Outputs include `metrics.json`, recommendation JSON, movie popularity CSV parts, and region/occupation behavior profile CSV parts. The CLI uses `local[*]` by default. Spark's first startup can take a short while.

## Input format

`ratings.csv` is required with columns `userId,movieId,rating,timestamp`; `users.csv` with `userId,age,occupation,region`; and `movies.csv` with `movieId,title,genres`. Extra columns are ignored. Ratings are deduplicated by `(userId, movieId)` and invalid/null/out-of-range rows are removed. Keep IDs numeric for Spark ALS and GraphX.

For a real MovieLens export, map its user/movie/rating/timestamp columns into these CSV headers. User attributes are optional for ALS but required by the behavior-profile join; use a complete `users.csv` with the stated schema. Ratings without matching users/movies remain eligible for ALS but do not contribute to profile-level analysis.

## GraphX job

The Scala GraphX app reads the same `ratings.csv` and writes `communities`, `popular_movies`, and `user_degrees` directories. Build an assembly jar with your standard Spark 3.5.1/sbt assembly setup, then submit it with the Spark core and GraphX dependencies supplied by the cluster. For example, once packaged as `movie-graphx-analysis_2.12-0.1.0.jar`:

```powershell
spark-submit --class MovieGraphAnalysis --master local[*] graphx/target/scala-2.12/movie-graphx-analysis_2.12-0.1.0.jar data/demo/ratings.csv outputs/graphx
```

Spark dependencies are marked `provided` in `graphx/build.sbt`; add the `sbt-assembly` plugin to `graphx/project/plugins.sbt` in an environment that builds the assembly jar, then run `sbt assembly` from `graphx/`.

## Evaluation and experiments

The seeded random split is rating-level, so it measures prediction on known users/items and may include cold-start rows; ALS drops cold-start predictions and reports `predictions_scored`. RMSE and MAE measure rating prediction error. Ranking metrics treat held-out ratings `>= 4.0` as relevant and compare them with each user's top-K ALS recommendations. Precision uses K as its denominator, including recommendation slots that overlap held-out relevance; report this definition when comparing systems.

Run unit tests:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Run a parameter sweep (masters, shuffle partitions, ALS rank, and repeats):

```powershell
$env:PYTHONPATH = "src"
python experiments/benchmark.py --data-dir data/demo --output-dir outputs/benchmark --masters 'local[2]' 'local[*]' --partitions 2 8 --ranks 8 16 --repeats 2
```

`benchmark.csv` captures wall time, Spark-reported fit/total time, RMSE, MAE, precision@K, and input size. For a meaningful scalability study, replace the demo data with increasing dataset sizes, keep hardware and Spark configuration fixed, warm up the runtime, run multiple repetitions, and report median/dispersion. Compare local runs only as a development baseline; cluster scalability requires measurements across worker counts and partitions. Treat Spark UI/stage metrics and executor resource utilization as companion evidence for diagnosing model performance.

## Distributed execution

Use the same module entrypoint on a Spark cluster, with package dependencies installed on executors and a shared filesystem path for input/output:

```powershell
spark-submit --master spark://HOST:7077 --py-files src/movie_recs.zip --conf spark.sql.shuffle.partitions=64 src/movie_recs/cli.py --data-dir hdfs:///datasets/movielens --output-dir hdfs:///results/movie-recs --rank 16 --iterations 10 --partitions 64
```

Package `src/movie_recs/` as `src/movie_recs.zip` before submitting, or distribute the source using your deployment's supported mechanism. Choose partition counts based on data size and cluster cores; do not assume a single fixed setting is optimal.
