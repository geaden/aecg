"""
Helper types and classes for Adaptive Erroneous Conditional Gradient.
"""

import enum
import time
from typing import Callable

import numpy as np
from abc import ABC, abstractmethod


# Gradient function typealias
Gradient = Callable[[int], np.ndarray]
History = tuple[np.ndarray, int]


_CHECK = True
_LOG_ENABLED = False
# Enables computing M value instead of relying passed value.
_COMPUTE_M = True


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


class Boundedness(enum.IntEnum):
    """
    Boundedness of the problem.
    """

    UNBOUNDED = 0
    BOUNDED = 1


class Algorithm(ABC):
    """
    Abstract algorithm.
    """

    _label: str
    _keep_history: bool
    _history: list[History]

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
    def history(self) -> list[History]:
        """Obtain history of algorithm.

        Returns:
            history
        """
        return self._history


class Objective(ABC):
    """
    Objective function for optimization by
    Adaptive Erroneous Conditional Gradient.
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

    _L_t: np.longdouble

    def __init__(self, L_0: np.complex128):
        self._L_t = L_0.astype(np.longdouble)
        self._is_adjust_needed = True

    def __call__(
        self,
        g_hat: np.ndarray,
        p: np.ndarray,
    ) -> float:
        """Calculate step size.

        Args:
            g: erroneous gradient
            p: search direction

        Returns:
            step size
        """
        return NotImplementedError()

    def adjust(self):
        """
        Adjust step size strategy.
        """
        self._L_t *= 0.5

    def adapt(self):
        """
        Adapt step size strategy.
        """
        self._L_t *= 2

    def is_adapted(
        self,
        f: np.ndarray,
        f_next: np.ndarray,
        g_hat: np.ndarray,
        p: np.ndarray,
    ) -> bool:
        """
        Check if step size strategy is adapted.
        This call is required in overridden classes.

        Args:
            objective: objective function
            f: value of the objective function at the current point
            f_next: value of the objective function at the next point
            g_hat: erroneous gradient
            p: search direction
        """
        raise NotImplementedError()


class MComputeMixin:
    """
    Mixing for computing $M$ value.
    """

    _M: np.longdouble

    def __init__(self):
        self._M = np.longdouble(0)

    def compute_M(self, gradient: np.ndarray):
        """
        Computes $M$ from the given inexact gradient value.

        Args:
          gradient: gradient value
        """
        if _COMPUTE_M:
            self._M = np.linalg.norm(gradient)


class SmoothnessStepSizeStrategy(StepSizeStrategy, MComputeMixin):
    """
    Implementation of |StepSizeStrategy| for |AECG|
    that depends on $L_t$.
    """

    _boundedness: Boundedness
    _epsilon: float
    _M: float
    _R: float

    def __init__(
        self,
        epsilon: float,
        L_0: np.longdouble,
        M: float,
        R: float,
        _boundedness: Boundedness,
    ):
        super().__init__(L_0)
        self._epsilon = epsilon
        self._M = M
        self._R = R
        self._L_0 = L_0
        self._is_adapted = True
        self._boundedness = _boundedness

    def __call__(self, g_hat: np.ndarray, p: np.ndarray) -> float:
        numerator = self._compute_numerator(g_hat, p)
        denominator = np.multiply(
            self._L_t, np.linalg.norm(p) ** 2, dtype=np.longdouble
        )
        return -numerator / denominator

    def is_adapted(
        self,
        f: np.ndarray,
        f_next: np.ndarray,
        g_hat: np.ndarray,
        p: np.ndarray,
    ) -> bool:
        numerator = np.pow(self._compute_numerator(g_hat, p), 2)
        denominator = np.multiply(
            np.multiply(2, self._L_t, dtype=np.longdouble),
            np.pow(np.linalg.norm(p), 2, dtype=np.longdouble),
            dtype=np.longdouble,
        )
        return f - f_next >= numerator / denominator

    def _compute_numerator(self, g_hat: np.ndarray, p: np.ndarray) -> np.longdouble:
        return self._boundedness * self._epsilon * self._M * self._R + np.dot(g_hat, p)


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
        error = (
            np.random.uniform(-1, 1, size=x.shape)
            * self.epsilon
            * np.abs(true_gradient)
        )
        erroneous_gradient = true_gradient + error

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
