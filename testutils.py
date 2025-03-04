import numpy as np
import unittest


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.eps = 0.1
        # Objective parameters
        self.A = np.random.randn(self.n, self.n)
        self.A *= np.sign(self.A)
        self.A /= self.A.sum(axis=0, keepdims=True)  # Normalize columns to sum to 1
        # Damping factor to simulate random surfing
        d = 0.85
        self.A = d * self.A + (1 - d) / self.n * np.ones((self.n, self.n))
        self.lower_bound = np.zeros(self.n)
        self.upper_bound = np.ones(self.n)
        self.w0 = np.random.uniform(self.lower_bound, self.upper_bound)
        self.L = np.linalg.eigvals(self.A).max().astype(np.longdouble)
        self.M = 10
        self.R = 1
        self.w0 = np.random.uniform(self.lower_bound, self.upper_bound)

    @staticmethod
    def main():
        np.random.seed(2025)
