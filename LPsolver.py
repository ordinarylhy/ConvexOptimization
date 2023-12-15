# linear programming solver with interior point method


# %%
import numpy as np


class NewtonSolver:
    """
    solves the centering problem
    minimize c.T @ x - np.sum(np.log(x))
    subject to Ax == b

    using Newton's method with backtracking line search
    the Newton step is determined via second-order approximation
    """
    def __init__(self, A, b, c, x0, tolerance=1e-6, alpha=0.01, beta=0.5):
        self.A = A
        self.b = b
        self.c = c
        self.x = x0
        self.tolerance = tolerance
        self.backtrack_alpha = alpha  # \alpha\in(0,0.5)
        self.backtrack_beta = beta  # \beta\in(0,1)
        self.Newton_decrement_history = []  # history of squared Newton decrements
        self.steps = 0
    
    def objective(self):
        c = self.c
        f = lambda x: c @ x - np.sum(np.log(x))
        return f
    
    def gradient(self):
        c = self.c 
        x = self.x 
        return c - 1 / x

    def hessian(self):
        x = self.x
        return np.diag(x ** -2)
    
    def direction(self):
        # w dual
        A = self.A 
        x = self.x 
        Hess_inv = np.diag(x ** 2)
        gradient = self.gradient()
        w = np.linalg.solve(A @ Hess_inv @ A.T, A @ Hess_inv @ -gradient)
        direction = Hess_inv @ -(A.T @ w + gradient)
        return direction, w
    
    def newton_decrement_square(self):
        direction, _ = self.direction() 
        return direction @ self.hessian() @ direction
    
    def backtrack(self):  
        t = 1
        f = self.objective()
        x = self.x
        alpha = self.backtrack_alpha
        beta = self.backtrack_beta
        direction, _ = self.direction()
        while np.any(x + t * direction <= 1e-10):
            t *= beta
        while f(x + t * direction) > f(x) + alpha * t * self.gradient() @ direction:
            t *= beta
        return t

    def solve(self):
        x = self.x
        while self.newton_decrement_square() / 2 > self.tolerance:
            direction, _ = self.direction()
            t = self.backtrack()
            x = x + t * direction
            self.x = x
            self.Newton_decrement_history.append(self.newton_decrement_square() / 2)
            self.steps += 1
        _, dual = self.direction()
        return x, dual


# %%
class LPsolver_startpoint():
    """
    solves the linear program given strictly feasible starting point x0
    minimize c.T @ x
    subject to
    Ax == b
    x >= 0

    using the barrier method
    """
    def __init__(self, A, b, c, x0, epsilon=1e-3, mu=3, t0=10):
        self.A = A
        self.b = b
        self.c = c
        self.x = x0
        self.epsilon = epsilon
        self.history = np.empty((2, 0)) # history of Newton steps and duality gap
        self.mu = mu  # mu > 1
        self.t = t0  # t0 > 0
        self.m = self.x.shape[0]  # number of constraints
    
    def barrier(self):
        while self.m / self.t >= self.epsilon: 
            self.t *= self.mu
            NewtonStep = NewtonSolver(self.A, self.b, self.t * self.c, self.x)
            self.x, _ = NewtonStep.solve()
            self.history = np.append(
                self.history,
                np.array([[NewtonStep.steps], [self.m / self.t]]),
                axis=1)
        return self.x


# %%
class LPsolver():
    """
    solves the linear program
    minimize c.T @ x
    subject to
    Ax == b
    x >= 0

    use basic phase I method to determine strict feasibility
    barrier method where descent is determined by
    Newton's method with backtracking line search
    """
    def __init__(self, A, b, c):
        self.A = A
        self.b = b
        self.c = c
        self.m = A.shape[1]
        
    def phaseI(self):  # basic phase I method
        x, *_ = np.linalg.lstsq(self.A, self.b, rcond=None)
        if np.min(x) > 0:
            return x
        else: 
            s = 1 - np.min(x)
            phase_I = LPsolver_startpoint(
                np.append(self.A, (-self.A @ np.ones(self.m)).reshape(-1, 1), axis=1),
                self.b - self.A @ np.ones(self.m),
                np.append(np.zeros(self.m), 1),
                np.append(x+s, s+1)
            )
            z = phase_I.solve()
            s = z[-1] - 1
            if s < 0:
                return z[:-1] - s
            else:
                raise Exception('infeasible')
    
    def solve(self):
        return LPsolver_startpoint(self.A, self.b, self.c, self.phaseI()).barrier()


class Newton_Infeasible:
    """
    solves the centering problem
    minimize c.T @ x - np.sum(np.log(x))
    subject to Ax == b

    using Newton's method with backtracking line search
    the Newton step is determined via second-order approximation
    """
    def __init__(self, A, b, c, x=None, nu=None, epsilon=1e-6, alpha=0.01, beta=0.5):
        self.A = A
        self.b = b
        self.c = c
        # initialize an infeasible primal start point (carefully such that the inverse of the Hessian is invertible)
        self.x = np.ones(A.shape[1]) if x is None else x
        # initialize an infeasible dual start point
        self.nu = np.zeros(A.shape[0]) if nu is None else nu
        self.epsilon = epsilon
        self.backtrack_alpha = alpha  # \alpha\in(0,0.5)
        self.backtrack_beta = beta  # \beta\in(0,1)
        self.steps = 0
        self.residue_hsitory = []
    
    def objective(self):
        c = self.c
        f = lambda x : c @ x - np.sum(np.log(x))
        return f
    
    def gradient(self):
        c = self.c 
        x = self.x 
        return c - 1 / x

    def hessian(self):
        x = self.x
        return np.diag(x ** -2)
    
    def direction(self):
        # w dual
        A = self.A 
        b = self.b
        x = self.x 
        Hess_inv = np.diag(x ** 2)
        gradient = self.gradient()
        w = np.linalg.solve(-A @ Hess_inv @ A.T, A @ Hess_inv @ gradient - (A @ x - b))
        delta_x = Hess_inv @ -(A.T @ w + gradient)
        delta_nu = w - self.nu
        return delta_x, delta_nu
    
    def residue_norm(self, x, nu):
        r_pri = self.A @ x - self.b 
        r_dual = self.c - 1 / x + self.A.T @ nu 
        return np.linalg.norm(np.append(r_pri, r_dual), 2)
    
    def backtrack(self):
        t = 1
        f = self.objective()
        x = self.x
        nu = self.nu
        alpha = self.backtrack_alpha
        beta = self.backtrack_beta
        delta_x, delta_nu = self.direction()
        while np.any(x + t * delta_x <= 0):
            t *= beta
        while self.residue_norm(x + t * delta_x, nu + t * delta_nu) > (1 - alpha * t) * self.residue_norm(x, nu):
            t *= beta
        self.x  = x + t * delta_x
        self.nu = nu + t * delta_nu

    def solve(self):
        self.steps = 0
        while np.max(np.abs(self.A @ self.x - self.b)) > 1e-4 or self.residue_norm(self.x, self.nu) > self.epsilon :
            self.backtrack()
            self.steps += 1
            self.residue_hsitory.append(self.residue_norm(self.x, self.nu))
        return self.x, self.nu


class LPsolver_Infeasible():
    """
    solves the linear program with infeasible start Newton method
    minimize c.T @ x
    subject to
    Ax == b
    x >= 0
    """
    def __init__(self, A, b, c, epsilon=1e-3, mu=3, t0=10):
        self.A = A
        self.b = b
        self.c = c
        self.x = None
        self.nu = None
        self.epsilon = epsilon
        self.history = np.empty((2, 0))  # history of Newton steps and duality gap
        self.mu = mu  # mu > 1
        self.t = t0  # t0 > 0
        self.m = self.A.shape[1]  # number of constraints
    
    def barrier(self):
        while self.m / self.t >= self.epsilon: 
            self.t *= self.mu
            NewtonStep = Newton_Infeasible(self.A, self.b, self.t * self.c, self.x, self.nu)
            self.x, self.nu = NewtonStep.solve()
            self.history = np.append(
                self.history,
                np.array([[NewtonStep.steps], [self.m / self.t]]),
                axis=1)
            x_star = self.x
            lambda_star = 1 / self.t / self.x  
            nu_star = self.nu / self.t
        return x_star, lambda_star, nu_star 