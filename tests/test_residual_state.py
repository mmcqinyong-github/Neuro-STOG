"""残差状态表示学习的合同测试（对照 C1 臂：学习更新 vs E1 调参 EMA vs E2 元学习率）。

性能方向假设由 20 种子研究裁决，不写入单元测试。
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from residual_state import (
    E2Cell,
    ResidualConfig,
    StateCell,
    fit_dlinear,
    run_c1,
    train_cell,
    train_e2,
)


class CellContractTest(unittest.TestCase):
    def test_state_norm_is_contractive_bounded(self):
        rng = np.random.default_rng(0)
        cfg = ResidualConfig(seed=1, n_vars=3, m=2)
        cell = StateCell(cfg.n_vars, cfg.m, rng)
        s = np.zeros(cfg.n_vars * cfg.m)
        for _ in range(500):
            s = cell.step(s, 50.0 * rng.normal(size=cfg.n_vars))
        self.assertLessEqual(float(np.abs(s).max()), 1.0 + 1e-9)

    def test_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(3)
        n_vars, m, T = 1, 1, 10
        cell = StateCell(n_vars, m, rng)
        rs = rng.normal(size=(T, n_vars))
        ys = rng.normal(size=(T, n_vars))
        loss, grads = cell.loss_and_grad(rs, ys)
        eps = 1e-6
        for name in ("W_r", "W_s", "a", "b", "P", "q"):
            param = cell.params[name]
            flat = param.ravel()
            gflat = grads[name].ravel()
            for idx in [0, flat.size - 1]:
                orig = flat[idx]
                flat[idx] = orig + eps
                lp, _ = cell.loss_and_grad(rs, ys)
                flat[idx] = orig - eps
                lm, _ = cell.loss_and_grad(rs, ys)
                flat[idx] = orig
                num = (lp - lm) / (2 * eps)
                den = max(abs(num), abs(gflat[idx]), 1e-8)
                self.assertLess(abs(num - gflat[idx]) / den, 1e-3,
                                f"{name}[{idx}] analytic={gflat[idx]:.6f} numeric={num:.6f}")

    def test_e2_update_is_elementwise(self):
        rng = np.random.default_rng(1)
        cell = E2Cell(2, rng)
        s = np.zeros(2)
        s = cell.step(s, np.array([5.0, 0.0]))
        self.assertEqual(s[1], 0.0)
        self.assertNotEqual(s[0], 0.0)


class FrozenContractTest(unittest.TestCase):
    def test_preqquential_evaluation_is_deterministic_and_frozen(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(100, 2))
        X = np.stack([x[t - 6 : t] for t in range(6, 90)])
        backbone = fit_dlinear(X, x[6:90], ridge=1e-4)
        original = {k: v.copy() for k, v in backbone.items()}
        cfg = ResidualConfig(seed=2, n_vars=2, n_train=70, n_calibrate=40,
                             n_test=10, lookback=6, train_epochs=2, restarts=1)
        r1 = run_c1(cfg, backbone=backbone)
        r2 = run_c1(cfg, backbone=backbone)
        self.assertEqual(r1.learned_wmse, r2.learned_wmse)
        for key, value in backbone.items():
            np.testing.assert_allclose(value, original[key], atol=1e-12)
        self.assertEqual(r1.n_predictions, 10)


class SmokeTest(unittest.TestCase):
    def test_c1_reports_finite_comparable_metrics(self):
        cfg = ResidualConfig(seed=101, n_vars=3, n_train=300, n_calibrate=300,
                             n_test=400, lookback=24, train_epochs=3, restarts=1)
        r = run_c1(cfg, backbone=None)
        for value in (r.base_wmse, r.learned_wmse, r.e1_wmse, r.e2_wmse):
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
        self.assertEqual(r.n_predictions, 400)


if __name__ == "__main__":
    unittest.main()
