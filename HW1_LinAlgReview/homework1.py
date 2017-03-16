""" homework1.py
    Nicholas S. Bradford
    CS 525 Deep Neural Networks
    1.18.2017

    HW1: numpy exercises
    Runs on Python 3.5
    Assume 0-based indexing.
"""

import numpy as np


def problem1 (A, B):
    """ Given matrices A and B, compute and return an expression for A + B. """
    return A + B

def problem2 (A, B, C):
    """ Given matrices A, B, and C, compute and return (AB - C)
        (i.e., right-multiply matrix A by matrixB, and then subtract C). 
        Use dot or numpy.dot. 
    """
    return np.dot(A, B) - C

def problem3 (A, B, C):
    """ Given matrices A, B, and C, return A  B + C>, where  represents the element-wise (Hadamard)
        product and > represents matrix transpose. In numpy, the element-wise product is obtained simply
        with *. 
    """
    return A * B + np.transpose(C)

def problem4 (x, y):
    """ Given column vectors x and y, compute the inner product of x and y (i.e., x>y). """
    return np.dot(np.transpose(x), y)

def problem5 (A):
    """ Given matrix A, return a matrix with the same dimensions as A but that contains all zeros. 
        Use numpy.zeros. 
    """
    return np.zeros(A.shape)

def problem6 (A):
    """ Given matrix A, return a vector with the same number of rows as A but that contains all ones. 
        Use numpy.ones. 
    """
    return np.ones(A.shape[0])

def problem7 (A):
    """ Given invertible matrix A, compute A-1 """
    return np.linalg.inv(A)

def problem8 (A, x):
    """ Given square matrix A and column vector x, use numpy.linalg.solve to compute A-1x. 
        If Ax = b, then x = A-1b. If Ab = x, then b = A-1x
    """
    return np.linalg.solve(A, x)

def problem9 (A, x):
    """ Given square matrix A and row vector x, use numpy.linalg.solve to compute xA-1. 
        Hint: AB = (B^T A^T)^T.
        We also know that the inverse of a transpose == the tranpose of the inverse,
            so solve(A^T, b^T) = (A^T)-1 b^T = (A-1)^T b^T = (bA-1)^T
    """ 
    return np.transpose(np.linalg.solve(np.transpose(A), np.transpose(x)))

def problem10 (A, alpha):
    """ Given square matrix A and (scalar) alpha, compute A + alpha I, where I is the identity matrix 
        with the same dimensions as A. Use numpy.eye.
    """
    return A + np.eye(A.shape[0]) * alpha

def problem11 (A, i, j):
    """ Given matrix A and integers i,j, return the jth column of the ith row of A, i.e., Aij."""
    return A[i, j]

def problem12 (A, i):
    """ Given matrix A and integer i, return the sum of all the entries in the ith row, 
        i.e.,  SUMj Aij. Do not use a loop, which in Python is very slow. 
        Instead use the numpy.sum function.
    """ 
    return np.sum(A[i, :])

def problem13 (A, c, d):
    """ Given matrix A and scalars c, d, compute the arithmetic mean over all entries of A 
        that are between c and d (inclusive). 
        In other words, if S = {(i,j) : c <= Aij <= d}, then compute (1/S) * SUM Aij
        Use numpy.nonzero along with numpy.mean.
    """
    mask = (A >= c) & (A <= d)
    return np.mean(A[np.nonzero(mask)])

def problem14 (A, k):
    """ Given an (n x n) matrix A and integer k, return an (n x k) matrix containing the 
        right-eigenvectors of A corresponding to the k largest eigenvalues of A. 
        Use numpy.linalg.eig to compute eigenvectors.
    """
    vals, vecs = np.linalg.eig(A)
    mask = np.argsort(vals)[-k:] # get last k elements
    return vecs[:, mask[::-1]] # reverse to get biggest elements first


def problem15 (x, k, m, s):
    """ Given a n-dimensional column vector x, an integer k, and positive scalars m, s, 
            return an (n x k) matrix, each of whose columns is a sample from multidimensional 
            Gaussian distribution N (x + mz, sI), where z is an n-dimensional column vector 
            containing all ones and I is the identity matrix. 
        Use either numpy.random.multivariate normal or numpy.random.randn.
    """ 
    n = x .shape[0]
    z = np.ones(n)
    i = np.eye(n)
    mean = x + (m * z)
    cov_matrix = s * i
    sample = np.random.multivariate_normal(mean, cov_matrix, k) 
    return np.transpose(sample) # make N x K matrix



# import unittest

# class HomeworkTest(unittest.TestCase):

#     A = np.arange(4).reshape((2,2))
#     B = np.arange(4, 8).reshape((2,2))
#     C = np.arange(8, 12).reshape((2,2))
#     x = np.arange(5)
#     y = np.arange(5, 10)
#     large = np.arange(25).reshape((5,5))
#     rect = np.arange(12).reshape((3, 4))
#     inv = np.array([[4, 3], [1, 1]])
#     row = np.arange(2)
#     large = np.array([[1, 2, 1], [6, -1, 0], [-1, -2, -1]])

#     def test_problems(self):
#         print(problem1(self.A, self.B))
#         print(problem2(self.A, self.B, self.C))
#         print(problem3(self.A, self.B, self.C))
#         print(problem4(self.x, self.y))
#         print(problem5(self.A))
#         print(problem6(self.rect))
#         print(problem7(self.inv))
#         print(problem8(self.inv, np.arange(2)))
#         print(problem9(self.inv, self.row))
#         print(problem10(self.A, 100))
#         self.assertEqual(1, problem11(self.A, 0, 1))
#         self.assertEqual(8 + 9 + 10 + 11, problem12(self.rect, 2))
#         self.assertEqual(1.5, problem13(self.rect, 0, 3))
#         print(problem14(self.large, 2))
#         print(problem15(self.x, k=3, m=7, s=5))

# if __name__ == '__main__':
#     unittest.main()
