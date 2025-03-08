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
    MComputeMixin,
    check,
)


class ECG(Algorithm, MComputeMixin):
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
    _delta: list[float]

    def __init__(
        self,
        label: str,
        max_iterations: int,
        objective: Objective,
        eo: ErroneousOracle,
        lmo: LinearMinimizationOracle,
        epsilon: float,
        M: float,
        R: float,
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

            self.compute_M(self._objective.gradient(w))
            L_t = max(self._compute_Lt(t, w, w_next, g_hat, p), np.longlong(0))
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
    ) -> np.longlong:
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

    def solve_delta(self):
        r"""
        Solve using erroneous conditional gradient with delta.

        \delta^t := f^t - f^*
        \delta^{t+1} \leqslant \frac{t}{t + 2}\delta^t +
        \frac{2(1 + \beta)\varepsilon M R}{t + 2} + \frac{4 L_t R^2}{(t + 2)^2}
        """
        check(len(self._history) > 0, "Enable history to solve for delta")

        dt = self._objective(self._history[0][0]) - self._objective(
            self._history[-1][0]
        )

        self._delta = [dt]
        for t in range(self._max_iterations):
            self.compute_M(self._history[t][0])
            delta_t = (
                t / (t + 2) * self._delta[t]
                + 2
                * (1 + self._boundedness)
                * self._epsilon
                * self._M
                * self._R
                / (t + 2)
                + self._Lt[t] * self._R**2 / (t + 2) ** 2
            )

            self._delta.append(delta_t)

    @property
    def Lt(self) -> list[float]:
        """
        History of $L_t$.

        Returns:
            History of $L_t$.
        """
        return self._Lt

    @property
    def delta(self) -> list[float]:
        r"""
        History of $\delta_t$.

        Returns:
            History of $\delta_t$
        """
        return self._delta
