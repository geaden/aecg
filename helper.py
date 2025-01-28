"""
Helper types and classes for Adaptive Erroneous Conditional Gradient.
"""

import time
from typing import Callable

import numpy as np
from abc import ABC, abstractmethod


# Gradient function typealias
type Gradient = Callable[[int], np.ndarray]


_CHECK = True
_LOG_ENABLED = False


def log(message: str):
    if _LOG_ENABLED:
        print(message)


def check(condition: bool, message: str):
    """
    Debug check.

    Args:
        condition: condition to check.
        message: message to print if condition is not met.
    """
    if _CHECK:
        assert condition, message
        return

    if not condition:
        print(message)


class Algorithm(ABC):
    """
    Abstract algorithm.
    """

    _label: str
    _keep_history: bool
    _history: list[tuple[np.ndarray, int]]

    def __init__(self, label: str, keep_history: bool = False):
        """
        Initializes new algorithm.

        Args:
            label: label of an algorithm.
            keep_history: if history of each iteration should be kept.
        """
        self._label = label
        self._keep_history = keep_history
        self._history = []

    @property
    def label(self) -> str:
        """Obtain label of an algorithm.

        Returns:
            label
        """
        return self._label

    def solve(self, w0: np.ndarray) -> np.ndarray:
        """Solve the problem.

        Args:
            w0: initial point

        Returns:
            solution
        """
        return NotImplementedError()

    def _track_history(self, w: np.ndarray) -> None:
        """Track history of algorithm.

        Args:
            w: point
        """
        if self._keep_history:
            self._history.append((w, time.time()))

    @property
    def history(self) -> list[tuple[int, np.ndarray]]:
        """Obtain history of algorithm.

        Returns:
            history
        """
        return self._history


class Objective(ABC):
    """
    Objective function for Adaptive Erroneous Conditional Gradient.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Calculate function value at a given point.

        Args:
            x: point

        Returns:
            function value
        """
        return NotImplementedError()

    @property
    def gradient(self) -> np.ndarray:
        """Obtain callable to cacluate gradient of objective function.

        Returns:
            gradient callable
        """
        return NotImplementedError()


class PageRankObjective(Objective):
    """
    PageRank function.

    .. math::
        f(x) = \\frac{1}{2}\\left\'lVert{Ax}\\right\\rVert_2^2
    """

    def __init__(self, A: np.ndarray) -> None:
        self.A = A

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * np.linalg.norm(self.A @ x) ** 2

    @property
    def gradient(self) -> Gradient:
        """
        Returns:
            A callable to compute a gradient at given point:

            $$
            \nab f(x) = A^T A x.
            $$
        """

        def f(x: np.ndarray) -> np.ndarray:
            return np.dot(self.A.T, np.dot(self.A, x))

        return f


class StepSizeStrategy(ABC):
    """
    Step size strategy for Adaptive Erroneous Conditional Gradient.
    """

    def __call__(
        self,
        t: int,
        g_hat: np.ndarray,
        p: np.ndarray,
    ) -> float:
        """Calculate step size.

        Args:
            t: iteration number
            g: erroneous gradient
            p: search direction

        Returns:
            step size
        """
        return NotImplementedError()

    def is_stop_criterion_reached(
        self,
        t: int,
        objective: Objective,
        w: np.ndarray,
        w_next: np.ndarray,
        f_opt: np.ndarray,
        g_hat: np.ndarray,
        p: np.ndarray,
    ) -> bool:
        """Check if stop criterion is reached.

        Args:
            t: iteration number
            w: current point to calculate objective value at
            w_next: next obtained point to calculate objective value at
            f_opt: approximation of optimal solution
            g_hat: erroneous gradient
            p: search direction

        Returns:
            True if stop criterion is reached, False otherwise
        """
        return NotImplementedError


class DecayingStepSizeStrategy(StepSizeStrategy):
    """
    Decaying step size strategy for Adaptive Erroneous Conditional Gradient.
    """

    _epsilon: float
    _Lt: float

    def __init__(self, epsilon: float):
        """
        Initialize decaying step size strategy.

        Args:
            epsilon: relative error
        """
        self._epsilon = epsilon

    def __call__(self, t, g_hat: np.ndarray, p: np.ndarray) -> float:
        return min(1, 2 / (t + 2))

    def is_stop_criterion_reached(
        self,
        t: int,
        objective: Objective,
        w: np.ndarray,
        w_next: np.ndarray,
        f_opt: np.ndarray,
        g_hat: np.ndarray,
        p: np.ndarray,
    ) -> bool:
        M = np.linalg.norm(objective.gradient(w))
        R = np.linalg.norm(p)
        rhs = self._epsilon * M * R + (4 * self._Lt * np.square(R) ** 2) / (
            (t + 2) ** 2
        )
        if objective(w_next) - f_opt <= rhs:
            print(f"Converged at iteration {t + 1}")
            return True
        return False


class ConstantDecayingStepSizeStrategy(DecayingStepSizeStrategy):
    """
    Constant decaying step size strategy for Adaptive Erroneous Conditional Gradient.
    """

    def __init__(self, epsilon: float, L: float):
        """
        Initialize constant decaying step size strategy.

        Args:
            epsilon: relative error
            L: c Lipschitz-gradient constant
        """
        super().__init__(epsilon)
        self._Lt = L


class DynamicDecayingStepSizeStrategy(DecayingStepSizeStrategy):
    """
    Dynamic decaying step size strategy for Adaptive Erroneous Conditional Gradient.
    """

    def __init__(self, epsilon):
        super().__init__(epsilon)
        self._Lt = 0

    def is_stop_criterion_reached(
        self, t, objective, w, w_next, f_opt, g_hat, p
    ) -> bool:
        M = np.linalg.norm(objective.gradient(w))
        R = np.linalg.norm(p)

        if R == 0:
            return False

        eta_t = self(t, g_hat, p)
        Lt = (
            objective(w_next)
            - objective(w)
            - eta_t * np.dot(g_hat, p)
            - eta_t * self._epsilon * M * R
        ) / (eta_t**2 * np.square(R))
        self._Lt = max(self._Lt, Lt)
        log(f"Lt = {self._Lt}")
        rhs = (
            t / (t + 2) * (objective(w) - f_opt)
            + 4 * self._epsilon * M * R / (t + 2)
            + 4 * self._Lt * np.square(R) / (t + 2) ** 2
        )
        if objective(w_next) - f_opt <= rhs:
            print(f"Converged at iteration {t + 1}")
            return True
        return False


class ErroneousOracle(ABC):
    """
    Abstract class for erroneous oracle.
    """

    def __init__(self, epsilon: float) -> None:
        """
        Construct erroneous oracle.

        Args:
            epsilon: relative error.
        """
        self._epsilon = epsilon

    @property
    def epsilon(self) -> float:
        """
        Returns:
            float: relative error that the oracle was instantiated with.
        """
        return self._epsilon

    @abstractmethod
    def __call__(self, gradient: Gradient, x: np.ndarray) -> np.ndarray:
        """
        Args:
            gradient: gradient of function.
            x: value to calculate erroneous gradient for.

        Returns:
            value of a erroneous gradient.
        """
        raise NotImplementedError()


class CWEOracle(ErroneousOracle):
    """Coordinate-wise implementation of |ErroneousOracle|."""

    def __call__(self, gradient: Gradient, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: value to calculate erroneous gradient for.

        Returns:
            value of a coordinate-wise erroneous gradient.
        """
        self._ensure_relative_error_bounds()
        true_gradient = gradient(x)
        erroneous_gradient = np.zeros_like(true_gradient)
        for i in range(len(true_gradient)):
            gradient_i = true_gradient[i]
            error = np.random.uniform(1 - self.epsilon, 1 + self.epsilon)
            # Relative error with sign preservation
            erroneous_gradient[i] = np.sign(gradient_i) * abs(gradient_i) * error

        check(
            self._is_valid(true_gradient, erroneous_gradient),
            "Properties of CWEO are violated.",
        )

        return erroneous_gradient

    def _is_valid(self, gradient: np.ndarray, erroneous_gradient: np.ndarray) -> bool:
        """
        Verifies properties of CWEO.

        Args:
            gradient: gradient of the function.
            erroneous_gradient: value of erroneous gradient

        Returns:
            If CWEO properties are violated or not.
        """

        # Check sign preservation
        for i in range(len(gradient)):
            if np.sign(erroneous_gradient[i]) != np.sign(gradient[i]):
                print(f"Sign preservation property failed at {i}")
                return False

        # Check relative coordinate-wise error
        for i in range(len(gradient)):
            abs_gradient = abs(gradient[i])
            abs_erroneous_gradient = abs(erroneous_gradient[i])
            if not (
                (1 - self.epsilon) * abs_gradient
                <= abs_erroneous_gradient
                <= (1 + self.epsilon) * abs_gradient
            ):
                print(f"Relative coordinate wise error condition violated at {i}")
                return False

            if not (
                1 / (1 + self.epsilon) * abs_erroneous_gradient
                <= abs_gradient
                <= 1 / (1 - self.epsilon) * abs_gradient
            ):
                print(f"Relative error bounds violated at {i}")
                return False

        return True

    def _ensure_relative_error_bounds(self):
        check(
            0 <= self.epsilon < 0.5,
            r"$\epsilon = {} \not\in [0; 0.5)$".format(self.epsilon),
        )


class LinearMinimizationOracle(ABC):
    """
    Abstract linear minimization oracle to solve
    optimization problem:

    $$
    LMO(d) := argmin_s{<d,z> : z C}
    $$
    """

    @abstractmethod
    def __call__(self, gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class BoxSetLMO(LinearMinimizationOracle):
    """
    Linear minimization oracle on a box set.
    """

    def __init__(self, lower_bound: np.ndarray, upper_bound: np.ndarray):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def __call__(self, gradient: np.ndarray) -> np.ndarray:
        """
        Solution to optimization problem.

        Args:
            gradient: $\nab_f(x)$

        Returns:
            solution of optimization problem on a box set.
        """
        return np.where(gradient > 0, self._lower_bound, self._upper_bound)
