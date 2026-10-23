import unittest

from movie_recs.metrics import ranking_metrics, regression_metrics


class MetricsTests(unittest.TestCase):
    def test_regression_metrics(self):
        result = regression_metrics([(3.0, 4.0), (5.0, 4.0)])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["mae"], 1.0)
        self.assertEqual(result["rmse"], 1.0)

    def test_regression_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            regression_metrics([])

    def test_ranking_metrics(self):
        result = ranking_metrics([10, 20, 30], [20, 30, 40], k=3)
        self.assertAlmostEqual(result["precision_at_k"], 2 / 3)
        self.assertAlmostEqual(result["recall_at_k"], 2 / 3)
        self.assertEqual(result["hit"], 1)

    def test_ranking_empty_relevance_and_invalid_k(self):
        self.assertEqual(ranking_metrics([1], [], 3)["hit"], 0)
        with self.assertRaises(ValueError):
            ranking_metrics([1], [1], 0)


if __name__ == "__main__":
    unittest.main()
