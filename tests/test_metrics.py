import unittest

import numpy as np
import pandas as pd

from steplens.data import CAT_KEYS
from steplens.metrics import binary_metrics, confirm_slices, normalize_scores, per_category, slice_table, wilson


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    env = rng.choice([1, 2], n)
    has_err = rng.random(n) < 0.4
    # the model misses most errors in kitchen 2 and few in kitchen 1
    miss_p = np.where(env == 2, 0.8, 0.2)
    pred = np.where(has_err, rng.random(n) > miss_p, rng.random(n) < 0.1)
    df = pd.DataFrame(dict(has_errors=has_err, environment_id=env, person_id=rng.choice([1, 2, 3], n),
                           activity_name=rng.choice(["A", "B"], n), duration_bucket="30-60s", position_bucket="Q1"))
    for k in CAT_KEYS:
        df[f"err_{k}"] = False
    df.loc[has_err, "err_order"] = True
    return df, pd.Series(pred, index=df.index)


class MetricsTest(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.278, places=2)
        lo, hi = wilson(5, 10)
        self.assertAlmostEqual(0.5 - lo, hi - 0.5, places=6)
        self.assertTrue(np.isnan(wilson(0, 0)[0]))

    def test_binary_metrics(self):
        m = binary_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.5)
        self.assertAlmostEqual(m["false_alarm"], 0.5)
        self.assertAlmostEqual(m["auc"], 0.75)

    def test_normalize_scores(self):
        s = normalize_scores(pd.Series([0.2, 0.4, 0.6]))
        self.assertEqual(list(s.round(3)), [0.0, 0.5, 1.0])
        self.assertEqual(list(normalize_scores(pd.Series([0.3, 0.3]))), [0.0, 0.0])

    def test_per_category(self):
        df, pred = synthetic()
        rows = {r["key"]: r for r in per_category(df, pred)}
        self.assertEqual(rows["order"]["n"], int(df.has_errors.sum()))
        self.assertTrue(np.isnan(rows["timing"]["recall"]))
        self.assertAlmostEqual(rows["normal"]["recall"], pred[~df.has_errors].mean())

    def test_slice_table_finds_the_bad_kitchen(self):
        df, pred = synthetic()
        tab = slice_table(df, pred, min_size=10, min_errors=10, max_order=1, include_categories=False)
        self.assertEqual(tab.iloc[0]["slice"], "environment_id=2")
        self.assertGreater(tab.iloc[0]["z_miss"], 3)
        self.assertTrue(0 <= tab.iloc[0]["miss_lo"] <= tab.iloc[0]["miss_rate"] <= tab.iloc[0]["miss_hi"] <= 1)
        # Wilson interval columns exist and the miss count matches the rate
        row = tab.iloc[0]
        self.assertAlmostEqual(row["n_missed"] / row["n_error"], row["miss_rate"])

    def test_confirm_slices_on_same_data_reproduces_rates(self):
        df, pred = synthetic()
        tab = slice_table(df, pred, min_size=10, min_errors=10, max_order=2, include_categories=True).head(20)
        out, overall = confirm_slices(df, pred, tab)
        self.assertAlmostEqual(overall, 1 - pred[df.has_errors].mean())
        for _, r in out.iterrows():
            self.assertEqual(r["confirm_n_error"], r["n_error"])
            self.assertAlmostEqual(r["confirm_miss_rate"], r["miss_rate"])


if __name__ == "__main__":
    unittest.main()
