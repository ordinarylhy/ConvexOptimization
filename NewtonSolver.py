"""
implements the Newton method to find the minimum of a convex function f
uses torch.autograd to evaluate the gradient and hessian
due to numerical reasons, need to try a few different starting point x_init to find the global minimum
"""



import numpy as np
import torch
from torch.autograd import grad
from torch.autograd.functional import hessian


class NewtonSolver:
    def __init__(self, f, x_init, tolerance=1e-4, alpha=0.01, beta=0.5):
        # f is the function to minimize, x_init is the initial point
        self.f = f
        self.x = torch.tensor(list(x_init), requires_grad=True, dtype=torch.float32)
        self.tolerance = tolerance
        self.history = []
        self.backtrack_alpha = alpha  # \alpha\in(0,0.5)
        self.backtrack_beta = beta  # \beta\in(0,1)

    def gradient(self):
        f = self.f
        x = self.x
        return grad(f(x), x)[0]

    def hess(self):
        f = self.f
        x = self.x
        return hessian(f, x)

    def direction(self):
        return -torch.linalg.solve(self.hess(), self.gradient())

    def newton_decrement_square(self):
        return self.gradient() @ torch.linalg.solve(self.hess(), self.gradient())

    def backtrack(self):  # \alpha\in(0,0.5), \beta\in(0,1)
        t = torch.tensor([1.0])
        f = self.f
        x = self.x
        alpha = self.backtrack_alpha
        beta = self.backtrack_beta
        direction = self.direction()
        while f(x + t * direction) > f(x) + alpha * t * self.gradient() @ direction:
            t *= beta
        return t

    def solve(self):
        x = self.x
        while self.newton_decrement_square() / 2 > self.tolerance:
            direction = self.direction()
            t = self.backtrack()
            x = torch.tensor(np.array(x.detach()) + np.array(t * direction), requires_grad=True)
            self.x = x
            self.history.append(self.newton_decrement_square())
        return x
