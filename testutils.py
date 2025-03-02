import numpy as np
import unittest


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.eps = 0.1
        self.A = np.random.randn(self.n, self.n)
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        self.lower_bound = np.zeros(self.n)
        self.upper_bound = np.ones(self.n)
        self.w0 = np.random.uniform(self.lower_bound, self.upper_bound)
        self.L = np.linalg.eigvals(self.A).max()
        self.M = 10
        self.R = 1
        self.w0 = np.random.uniform(self.lower_bound, self.upper_bound)

    @staticmethod
    def main():
        np.random.seed(2025)
