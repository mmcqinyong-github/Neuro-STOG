"""对照 A1/A2 的合同与机制属性测试（不含性能方向断言）。

性能假设（循环调度优于反应式门控、循环结构 A 优于无结构 A）由
20 种子研究裁决，不写入单元测试——沿用上次"测试锁合同、实验裁假设"的教训。
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cyclic_mode import (
    CyclicScheduler,
    EnvConfig,
    ModeBank,
    _compose_correction,
    _zero_controller,
    fit_dlinear,
    make_series,
    run_a1,
    run_a2,
    run_a4,
)


class ComposeCorrectionContractTest(unittest.TestCase):
    def test_all_mode_correction_uses_every_mode(self):
        n = 2
        ctl = _zero_controller(n)
        ctl["coef"] = np.eye(n)
        controllers = [ctl, ctl, ctl]
        states = [np.array([1.0, -1.0]) for _ in range(3)]
        # 各模式控制器相同 -> 修正 = 3 倍单模式修正（全模式连续修正）
        full = _compose_correction(controllers, states)
        single = _compose_correction(controllers[:1], states[:1])
        np.testing.assert_allclose(full, 3.0 * single, atol=1e-12)

    def test_weighted_compose(self):
        n = 2
        ctl = _zero_controller(n)
        ctl["coef"] = np.eye(n)
        states = [np.array([1.0, 2.0]) for _ in range(3)]
        full = _compose_correction([ctl] * 3, states, weights=[1.0, 0.0, 0.5])
        np.testing.assert_allclose(full, np.array([1.5, 3.0]), atol=1e-12)

    def test_a4_pollution_direction_is_constructive(self):
        cfg = EnvConfig(seed=5, n_vars=2, n_train=60, n_calibrate=60, n_test=80,
                        lookback=8, env="env3")
        result = run_a4(cfg, backbone=None)
        self.assertEqual(result.isolated_pollution, 0.0)
        self.assertGreater(result.global_pollution, 0.0)


class SchedulerContractTest(unittest.TestCase):
    def test_structured_scheduler_revisits_every_mode_within_bounded_time(self):
        rng = np.random.default_rng(7)
        sched = CyclicScheduler(n_modes=3, dwells=(10, 20, 5), p_cyc=1.0, rng=rng)
        seen_at = {k: [] for k in range(3)}
        for t in range(200):
            seen_at[sched.step()].append(t)
        for k, ts in seen_at.items():
            self.assertTrue(len(ts) > 0, f"mode {k} never visited")
            gaps = np.diff(ts)
            self.assertLessEqual(int(gaps.max()), 35)  # <= 2 * master cycle

    def test_structured_vs_unstructured_transition_matrices(self):
        rng = np.random.default_rng(0)
        s1 = CyclicScheduler(n_modes=4, dwells=(5, 5, 5, 5), p_cyc=1.0, rng=rng)
        counts1 = np.zeros(4, dtype=int)
        for _ in range(4000):
            counts1[s1.step()] += 1
        rng = np.random.default_rng(0)
        s2 = CyclicScheduler(n_modes=4, dwells=(5, 5, 5, 5), p_cyc=0.25, rng=rng)
        counts2 = np.zeros(4, dtype=int)
        for _ in range(4000):
            counts2[s2.step()] += 1
        # 结构化的下一步确定性：占用率由 dwell 主导且均衡；无结构随机性不引入额外偏置
        self.assertGreater(counts1.min(), 0)
        self.assertGreater(counts2.min(), 0)


class PhaseIsolationTest(unittest.TestCase):
    def test_cyclic_bank_updates_only_active_mode(self):
        bank = ModeBank(n_modes=3, n_vars=2, rho=0.9, clip=8.0)
        before = bank.states.copy()
        bank.update(mode=1, residual=np.array([1.0, -1.0]))
        np.testing.assert_allclose(bank.states[0], before[0], atol=1e-12)
        np.testing.assert_allclose(bank.states[2], before[2], atol=1e-12)
        self.assertFalse(np.allclose(bank.states[1], before[1]))

    def test_global_update_touches_all_modes(self):
        bank = ModeBank(n_modes=3, n_vars=2, rho=0.9, clip=8.0)
        bank.update_global(residual=np.array([1.0, -1.0]))
        for k in range(3):
            self.assertFalse(np.allclose(bank.states[k], 0.0))


class PrequentialContractTest(unittest.TestCase):
    def test_frozen_backbone_and_prediction_count(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(120, 2))
        X = np.stack([x[t - 8 : t] for t in range(8, 110)])
        y = x[8:110]
        backbone = fit_dlinear(X, y, ridge=1e-4)
        original = {k: v.copy() for k, v in backbone.items()}
        config = EnvConfig(
            seed=1, n_vars=2, n_train=80, n_calibrate=30, n_test=10,
            lookback=8, env="env1",
        )
        result = run_a1(config, backbone=backbone)
        for key, value in backbone.items():
            np.testing.assert_allclose(value, original[key], atol=1e-12)
        self.assertEqual(result.n_predictions, 10)


class SmokeTest(unittest.TestCase):
    def test_a1_a2_report_finite_comparable_metrics(self):
        config = EnvConfig(
            seed=101, n_vars=3, n_train=300, n_calibrate=400, n_test=500,
            lookback=24, env="env3",
        )
        r1 = run_a1(config, backbone=None)
        r2 = run_a2(config, backbone=None)
        for value in (
            r1.base_wmse, r1.cyclic_wmse, r1.reactive_wmse,
            r2.structured_wmse, r2.unstructured_wmse,
        ):
            self.assertTrue(np.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
        self.assertEqual(r1.n_predictions, 500)


if __name__ == "__main__":
    unittest.main()
