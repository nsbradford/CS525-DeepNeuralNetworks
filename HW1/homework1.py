""" homework1.py
    Nicholas S. Bradford
    CS 525 Deep Neural Networks
    1.18.2017

    HW1: numpy exercises
"""

import unittest
import numpy as np


def problem1 (A, B):
    """ Given matrices A and B, compute and return an expression for A + B. [ 2 pts ]"""
    return A + B

def problem2 (A, B, C):
    """ Given matrices A, B, and C, compute and return (AB - C)
        (i.e., right-multiply matrix A by matrixB, and then subtract C). 
        Use dot or numpy.dot. [ 2 pts ]
    """
    return np.dot(A, B) - C

def problem3 (A, B, C):
    """ Given matrices A, B, and C, return A  B + C>, where  represents the element-wise (Hadamard)
        product and > represents matrix transpose. In numpy, the element-wise product is obtained simply
        with *. [ 2 pts ]
    """
    return A * B + np.transpose(C)

def problem4 (x, y):
    """ Given column vectors x and y, compute the inner product of x and y (i.e., x>y). [ 2 pts ]"""
    return np.dot(np.transpose(x), y)

def problem5 (A):
    """ Given matrix A, return a matrix with the same dimensions as A but that contains all zeros. 
        Use numpy.zeros. [ 2 pts ]
    """
    return np.zeros(A.shape)

def problem6 (A):
    pass

def problem7 (A):
    pass

def problem8 (A, x):
    pass

def problem9 (A, x):
    pass

def problem10 (A, alpha):
    pass

def problem11 (A, i, j):
    pass

def problem12 (A, i):
    pass

def problem13 (A, c, d):
    pass

def problem14 (A, k):
    pass

def problem15 (x, k, m, s):
    pass


class HomeworkTest(unittest.TestCase):

    A = np.arange(4).reshape((2,2))
    B = np.arange(4, 8).reshape((2,2))
    C = np.arange(8, 12).reshape((2,2))
    x = np.arange(5)
    y = np.arange(5, 10)

    def test_problem1(self):
        print problem1(self.A, self.B)

    def test_problem2(self):
        print problem2(self.A, self.B, self.C)

    def test_problem3(self):
        print problem3(self.A, self.B, self.C)

    def test_problem4(self):
        print problem4(self.x, self.y)

    def test_problem5(self):
        print problem5(self.A)

    def test_problem6(self):
        pass

    def test_problem7(self):
        pass

    def test_problem8(self):
        pass

    def test_problem9(self):
        pass

    def test_problem10(self):
        pass

    def test_problem11(self):
        pass

    def test_problem12(self):
        pass

    def test_problem13(self):
        pass

    def test_problem14(self):
        pass

    def test_problem15(self):
        pass


if __name__ == '__main__':
    unittest.main()