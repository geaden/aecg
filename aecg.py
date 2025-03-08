"""
Implementation of Adaptive Erroneous Conditional Gradient.
"""

import numpy as np

from helper import (
    Algorithm,
    Objective,
    ErroneousOracle,
    LinearMinimizationOracle,
    StepSizeStrategy,
    log,
)


class AECG(Algorithm):
    """
    Implementation of Adaptive Erroneous Conditional Gradient.
    """

    _DELTA = 1e-4

    _objective: Objective
    _eo: ErroneousOracle
    _lmo: LinearMinimizationOracle
    _step_size: StepSizeStrategy

    def __init__(
        self,
        label: str,
        max_iterations: int,
        objective: Objective,
        eo: ErroneousOracle,
        lmo: LinearMinimizationOracle,
        step_size: StepSizeStrategy,
        keep_history=False,
    ):
        """ """
        super().__init__(label, keep_history)
        self._max_iterations = max_iterations
        self._objective = objective
        self._eo = eo
        self._lmo = lmo
        self._step_size = step_size

    def solve(self, w0: np.ndarray) -> np.ndarray:
        """
        Solve using erroneous conditional gradient.

        Args:
            w0: initial value.

        Returns:
            Optimal solution.
        """
        # Initial value
        w = w0.copy().astype(np.float64)
        self._track_history(w)
        condition = float("inf")
        for t in range(self._max_iterations):
            # Adjust step size
            self._step_size.adjust()

            # Obtain gradient value from ErroneousOracle
            g_hat = self._eo(self._objective.gradient, w)
            log(f"{g_hat=}")

            # Gradient direction using LMO
            p = self._lmo(g_hat) - w
            log(f"{p=}")

            while True:
                # Compute M value
                self._step_size.compute_M(self._objective.gradient(w))
                log(f"{self._step_size._M=}")

                # Choose step size $\eta_t$
                eta_t = self._step_size(g_hat, p)
                log(f"{eta_t=}")

                # Compute new value
                w_next = w + eta_t * p
                log(f"{w_next=}")

                f = self._objective(w)
                f_next = self._objective(w_next)

                if self._step_size.is_adapted(f, f_next, g_hat, p):
                    break

                self._step_size.adapt()

            w = w_next
            self._track_history(w)
            condition = min(
                abs(self._step_size._boundedness + np.dot(g_hat, p)), condition
            )
            if condition <= self._DELTA:
                break

        return w
