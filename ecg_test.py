import unittest

from ecg import ECG
from helper import Boundedness, BoxSetLMO, CWEOracle, PageRankObjective
from testutils import BaseTestCase


class TestErroneousConditionalGradient(BaseTestCase):
    def test_run_ecg(self):
        """Run ECG with the adaptive $L_t$"""
        for boundedness in Boundedness:
            with self.subTest(beta=boundedness):
                under_test = self._create_under_test(boundedness)

                result = under_test.solve(self.w0)

                self.assertIsNotNone(result)
                self.assertTrue(len(under_test.history) > 1)
                self.assertTrue(len(under_test.Lt) > 1)

    def test_solve_delta(self):
        """Run ECG with the adaptive $L_t$"""
        for boundedness in Boundedness:
            with self.subTest(beta=boundedness):
                under_test = self._create_under_test(boundedness)
                under_test.solve(self.w0)

                under_test.solve_delta()

                self.assertTrue(len(under_test.delta) > 1)

    def _create_under_test(self, boundedness: Boundedness) -> ECG:
        return ECG(
            label=r"Testing Erroneous Conditional Gradient",
            max_iterations=1000,
            objective=PageRankObjective(self.A),
            eo=CWEOracle(self.eps),
            lmo=BoxSetLMO(self.lower_bound, self.upper_bound),
            boundedness=boundedness,
            epsilon=self.eps,
            M=self.M,
            R=self.R,
            keep_history=True,
        )


if __name__ == "__main__":
    BaseTestCase.main()
    unittest.main()
