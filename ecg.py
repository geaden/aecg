"""
Implementation of  Erroneous Conditional Gradient with the adaptive L-smoothness
constant.
"""

import numpy as np

from helper import (
    Algorithm,
    Boundedness,
    ErroneousOracle,
    LinearMinimizationOracle,
    Objective,
    log,
)


class ECG(Algorithm):
    """
    Implementation of Erroneous Conditional Gradient.
    """

    _objective: Objective
    _eo: ErroneousOracle
    _lmo: LinearMinimizationOracle
    _boundedness: Boundedness
    _epsilon: float
    _M: float
    _R: float
    _Lt: list[float]

    def __init__(
        self,
        label: str,
        max_iterations: int,
        objective: Objective,
        eo: ErroneousOracle,
        lmo: LinearMinimizationOracle,
        epsilon: float,
        M: float = 1.0,
        R: float = 1.0,
        boundedness: Boundedness = Boundedness.UNBOUNDED,
        keep_history=False,
    ):
        """
        Initialize algorithm.

        Args:
            label: label of an algorithm.
            max_iterations: maximum number of iterations.
            objective: objective function.
            eo: erroneous oracle.
            lmo: linear minimization oracle.
            epsilon: relative error.
            M: constant M.
            R: radius in simplex unit.
            boundedness: boundedness of the problem.
            keep_history: if history of each iteration should be kept.
        """
        super().__init__(label, keep_history)
        self._max_iterations = max_iterations
        self._objective = objective
        self._eo = eo
        self._lmo = lmo
        self._epsilon = epsilon
        self._boundedness = boundedness
        self._M = M
        self._R = R
        self._Lt = []

    def solve(self, w0: np.ndarray):
        """
        Solve using erroneous conditional gradient.

        Args:
            w0: initial value.

        Returns:
            Optimal solution, value of $L_t$.
        """
        # Initial value
        w = w0.copy().astype(np.float64)

        self._track_history(w)

        for t in range(self._max_iterations):
            # Obtain inexact gradient value from ErroneousOracle
            g_hat = self._eo(self._objective.gradient, w)
            log(f"{g_hat=}")

            # Inexact gradient direction using LMO
            p = self._lmo(g_hat) - w
            log(f"{p=}")

            # Choose eta_t
            eta_t = 2 / (t + 2)
            log(f"{eta_t=}")

            # Compute new value
            w_next = w + eta_t * p
            log(f"{w_next=}")

            L_t = self._compute_Lt(t, w, w_next, g_hat, p)

            self._Lt.append(L_t)

            self._track_history(w_next)

            w = w_next

        log(self._history)
        return w

    def _compute_Lt(
        self,
        t: int,
        w: np.ndarray,
        w_next: np.ndarray,
        g_hat: np.ndarray,
        p: np.ndarray,
    ) -> float:
        """
        Compute $L_t$.

        Returns:
            $L_t$.
        """
        j = t + 2
        inexact_part = (
            np.dot(g_hat, p) + self._boundedness * self._epsilon * self._M * self._R
        )
        numerator = j * (
            j * (self._objective(w_next) - self._objective(w)) - 2 * inexact_part
        )
        denominator = 2 * np.linalg.norm(p) ** 2
        return numerator / denominator

    @property
    def Lt(self) -> list[float]:
        """
        History of $L_t$.

        Returns:
            History of $L_t$.
        """
        return self._Lt
