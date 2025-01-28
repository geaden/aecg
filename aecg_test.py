"""Unit tests of Adaptive Erroneous Conditional Gradient"""

import unittest

import numpy as np

from aecg import AECG
from helper import BoxSetLMO, CWEOracle, StepSizeStrategy, PageRankObjective
from helper import ConstantDecayingStepSizeStrategy, DynamicDecayingStepSizeStrategy


class ConstantDecayingStepSizeStrategyTest(unittest.TestCase):

    def setUp(self):
        self.under_test = ConstantDecayingStepSizeStrategy(0.1, 13)

    def test_step_size(self):
        eta = self.under_test(2, np.zeros(10), np.zeros(10))

        self.assertEqual(0.5, eta)


class DynamicDecayingStepSizeStrategyTest(unittest.TestCase):

    def setUp(self):
        self.under_test = DynamicDecayingStepSizeStrategy(0.1)

    def test_step_size(self):
        eta = self.under_test(2, np.zeros(10), np.zeros(10))

        self.assertEqual(0.5, eta)

    def test_is_stop_criterion_reached(self):
        np.random.seed(2025)
        self.n = 100
        self.eps = 0.1
        self.A = np.random.randn(self.n, self.n)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        self.under_test(0, np.zeros(self.n), np.zeros(self.n))

        self.assertFalse(
            self.under_test.is_stop_criterion_reached(
                t=0,
                objective=PageRankObjective(self.A),
                w=np.zeros(self.n),
                w_next=np.zeros(self.n),
                f_opt=64,
                g_hat=np.zeros(self.n),
                p=np.zeros(self.n),
            )
        )


class AECGTest(unittest.TestCase):
    """Adaptive Erroneous Conditional Gradient."""

    def setUp(self):
        self.n = 100
        self.eps = 0.1
        self.A = np.random.randn(self.n, self.n)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        self.lower_bound = np.zeros(self.n)
        self.upper_bound = np.ones(self.n)
        self.w0 = np.random.uniform(self.lower_bound, self.upper_bound)
        self.L = np.linalg.eigvals(self.A).max()
        self.w0 = np.random.uniform(self.lower_bound, self.upper_bound)

    def test_solve_algorithm_with_constant_decaying_step_size(self):
        """Test run iteration dependent with constant L."""
        under_test = self._create_under_test(
            ConstantDecayingStepSizeStrategy(self.eps, self.L)
        )

        result = under_test.solve(self.w0)

        self.assertIsNotNone(result)

    def test_solve_algorithm_with_dynamic_decaying_step_size(self):
        """Test run iteration dependent with constant L."""
        under_test = self._create_under_test(DynamicDecayingStepSizeStrategy(self.eps))

        result = under_test.solve(self.w0)

        self.assertIsNotNone(result)

    def _create_under_test(self, step_size: StepSizeStrategy) -> AECG:
        return AECG(
            label=r"Testing Adaptive Erroneous Conditional Gradient",
            max_iterations=1000,
            objective=PageRankObjective(self.A),
            eo=CWEOracle(self.eps),
            lmo=BoxSetLMO(self.lower_bound, self.upper_bound),
            step_size=step_size,
        )


if __name__ == "__main__":
    np.random.seed(2025)
    unittest.main()
