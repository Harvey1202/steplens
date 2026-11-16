import unittest

import numpy as np
import pandas as pd

from steplens.embedding import embedding_name, knn_agreement, markdown, step_occurrences


class EmbeddingTest(unittest.TestCase):
    def test_knn_agreement_separated_clusters(self):
        rng = np.random.default_rng(0)
        Z = np.vstack([rng.normal(0, 0.1, (50, 3)), rng.normal(10, 0.1, (50, 3))])
        r = knn_agreement(Z, np.repeat([0, 1], 50), k=5)
        self.assertEqual(r["agreement"], 1.0)
        self.assertAlmostEqual(r["chance"], 2 * 50 * 49 / (100 * 99))
        self.assertGreater(r["lift"], 1.9)

    def test_knn_agreement_random_labels_near_chance(self):
        rng = np.random.default_rng(1)
        Z = rng.normal(size=(600, 5))
        r = knn_agreement(Z, rng.integers(0, 4, 600), k=10)
        self.assertAlmostEqual(r["agreement"], r["chance"], delta=0.05)

    def test_knn_agreement_excludes_self_with_duplicates(self):
        Z = np.zeros((6, 2))  # all points identical: self must still be excluded
        r = knn_agreement(Z, np.array([0, 0, 0, 1, 1, 1]), k=5)
        self.assertEqual(r["k"], 5)
        self.assertAlmostEqual(r["agreement"], 2 / 5)

    def test_knn_agreement_tiny_input(self):
        self.assertTrue(np.isnan(knn_agreement(np.zeros((2, 2)), np.array([0, 1]))["agreement"]))

    def test_step_occurrences_order_and_missing(self):
        seg = pd.DataFrame(dict(recording_id=["r", "r", "r"], step_id=[1, 1, 2], ann_index=[3, 0, 1],
                                start_time=[50.0, 1.0, -1.0], end_time=[60.0, 9.0, -1.0], is_missing=[False, False, True]))
        occ = step_occurrences(seg)
        self.assertEqual(occ, {("r", 1): [(1.0, 9.0), (50.0, 60.0)]})

    def test_names_and_report(self):
        self.assertEqual(embedding_name("omnivore"), "omnivore")
        self.assertEqual(embedding_name("omnivore", center="recording"), "omnivore-center")
        self.assertEqual(embedding_name("imagebind", ("video", "audio")), "imagebind-video+audio")
        info = dict(name="omnivore", backbone="omnivore", center="none", n_steps=10, feature_dims=4, pca_dims=2, k=3,
                    neighbours=[dict(label="person", agreement=0.5, chance=0.25, lift=2.0)],
                    outcomes=[dict(model_id="m", n=5, missed_rate=0.2, agreement=0.6, chance=0.68, lift=0.88)])
        md = markdown(info)
        self.assertIn("| person | 50.0% | 25.0% | 2.0× |", md)
        self.assertIn("| m | 5 | 20.0% | 60.0% | 68.0% | 0.9× |", md)


if __name__ == "__main__":
    unittest.main()
