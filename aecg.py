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


class ECG(Algorithm):
    """
    Implementation of Erroneous Conditional Gradient.
    """

    _objective: Objective
    _eo: ErroneousOracle
    _lmo: LinearMinimizationOracle
    _step_size: StepSizeStrategy
    _is_bounded: bool
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
        keep_history=False,
        is_bounded: bool = False,
        M: float = 1.0,
        R: float = 1.0,
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
            keep_history: if history of each iteration should be kept.
            is_bounded: if value of $L_t$ is bounded.
            M: constant M.
            R: radius in simplex unit.
        """
        super().__init__(label, keep_history)
        self._max_iterations = max_iterations
        self._objective = objective
        self._eo = eo
        self._lmo = lmo
        self._epsilon = epsilon
        self._is_bounded = is_bounded
        self._M = M
        self._R = R
        self._Lt = []

    def solve(self, w0: np.ndarray):
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

        for t in range(self._max_iterations):
            # Obtain gradient value from ErroneousOracle
            g_hat = self._eo(self._objective.gradient, w)
            log(f"{g_hat=}")

            # Gradient direction using LMO
            p = self._lmo(g_hat) - w
            log(f"{p=}")

            # Choose eta_t
            eta_t = 2 / (t + 2)
            log(f"{eta_t=}")

            # Compute new value
            w_next = w + eta_t * p
            log(f"{w_next=}")

            add = (
                0
                if not self._is_bounded
                else -eta_t * self._epsilon * self._M * self._R
            )

            L_t = (
                2
                * (
                    self._objective(w_next)
                    - self._objective(w)
                    - eta_t * np.dot(g_hat, p)
                    + add
                )
                / (eta_t**2 * np.linalg.norm(p) ** 2)
            )

            self._Lt.append(L_t)

            self._track_history(w_next)

            w = w_next

        log(self._history)
        return w

    @property
    def Lt(self) -> list[float]:
        """
        Get history of $L_t$.

        Returns:
            History of $L_t$.
        """
        return self._Lt


class AECG(Algorithm):
    """
    Implementation of Adaptive Erroneous Conditional Gradient.
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

        for t in range(self._max_iterations):
            # Adjust step size
            self._step_size.adjust()
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

            if self._step_size.is_adapted(t, self._objective, w, w_next, g_hat, p):
                w = w_next
            else:
                # Adapt step size
                self._step_size.adapt()

            self._track_history(w)

        return w
