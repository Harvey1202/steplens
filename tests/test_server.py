import math
import unittest
from unittest import mock

import numpy as np

from steplens import server
from test_metrics import synthetic


class ServerHelperTests(unittest.TestCase):
    def test_finite_replaces_nan_and_inf(self):
        out = server._finite(dict(auc=float("nan"), f1=0.5, nested=[np.float64("inf"), 1], n=3))
        self.assertIsNone(out["auc"])
        self.assertEqual(out["f1"], 0.5)
        self.assertEqual(out["nested"], [None, 1])
        self.assertEqual(out["n"], 3)
        self.assertIsNone(server._finite(float("nan")))
        self.assertFalse(isinstance(server._finite(0.25), float) and math.isnan(server._finite(0.25)))

    def test_subset_applies_every_filter(self):
        df, _ = synthetic()
        df["activity_id"] = 1
        df["step_id"] = np.arange(len(df)) % 5
        df.loc[df.index[: len(df) // 2], "duration_bucket"] = "15-30s"
        df.loc[df.index[::3], "position_bucket"] = "Q4"
        with mock.patch.object(server.store, "scored_steps", return_value=df):
            sub = server._subset("m", "test", None, (), (), (2,), ("15-30s",), ("Q4",), (1,))
        expected = df[(df.environment_id == 2) & (df.duration_bucket == "15-30s")
                      & (df.position_bucket == "Q4") & (df.step_id == 1)]
        self.assertEqual(len(sub), len(expected))
        self.assertGreater(len(sub), 0)
        self.assertTrue(sub.index.equals(expected.index))


if __name__ == "__main__":
    unittest.main()
