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
    Implementation of Erroneous Conditional Gradient with Adaptive Step Size.
    """

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

        f_opt = self._objective(w)

        for t in range(self._max_iterations):
            # Obtain gradient value from ErroneousOracle
            g_hat = self._eo(self._objective.gradient, w)
            log(f"{g_hat=}")

            # Gradient direction using LMO
            p = self._lmo(g_hat) - w
            log(f"{p=}")

            # Choose step size $\eta \in [0; 1]$
            eta_t = self._step_size(t, g_hat, p)
            log(f"{eta_t=}")

            # Compute new value
            w_next = w + eta_t * p
            log(f"{w_next=}")

            # Find optimal solution.
            f_opt = min(f_opt, self._objective(w_next))
            log(f"{f_opt=}")

            if self._step_size.is_stop_criterion_reached(
                t, self._objective, w, w_next, f_opt, g_hat, p
            ):
                print(f"Converged at {t + 1}")
                break

            w = w_next

            self._track_history(w)

        return w
