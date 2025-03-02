"""Unit tests of Adaptive Erroneous Conditional Gradient"""

import unittest

import numpy as np

from aecg import AECG
from helper import (
    Boundedness,
    BoxSetLMO,
    CWEOracle,
    StepSizeStrategy,
    PageRankObjective,
)
from helper import SmoothnessStepSizeStrategy
from testutils import BaseTestCase


class AECGTest(BaseTestCase):
    def test_run_aecg(self):
        """Run AECG."""
        for boundedness in Boundedness:
            with self.subTest(beta=boundedness):
                under_test = self._create_under_test(
                    SmoothnessStepSizeStrategy(
                        self.eps, self.L, self.M, self.R, boundedness
                    )
                )

                result = under_test.solve(self.w0)

                self.assertIsNotNone(result)
                self.assertTrue(len(under_test.history) > 1)

    def _create_under_test(self, step_size: StepSizeStrategy) -> AECG:
        return AECG(
            label=r"Testing Adaptive Erroneous Conditional Gradient",
            max_iterations=1000,
            objective=PageRankObjective(self.A),
            eo=CWEOracle(self.eps),
            lmo=BoxSetLMO(self.lower_bound, self.upper_bound),
            step_size=step_size,
            keep_history=True,
        )


if __name__ == "__main__":
    BaseTestCase.main()
    unittest.main()
