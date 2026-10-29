"""Benchmark harness for the quasi-Newton notebook.

Re-implementation of the helper provided with the course, so that the
notebook runs on its own. ``test_solver`` runs an optimizer on three classic
test problems and plots its iterates and convergence.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


def _gaussian():
    """Non-convex: one minus an anisotropic Gaussian bump."""
    A = np.array([[1.0, 0.8], [0.8, 2.0]])

    def f(x):
        return 1.0 - np.exp(-0.5 * x @ A @ x)

    def grad(x):
        return np.exp(-0.5 * x @ A @ x) * (A @ x)

    def hess(x):
        e = np.exp(-0.5 * x @ A @ x)
        Ax = A @ x
        return e * (A - np.outer(Ax, Ax))

    return f, grad, hess, np.array([0.6, -0.6]), np.zeros(2)


def _quadratic():
    """Convex but badly conditioned (condition number 100), rotated axes."""
    theta = np.pi / 6
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = R @ np.diag([1.0, 100.0]) @ R.T

    def f(x):
        return 0.5 * x @ A @ x

    def grad(x):
        return A @ x

    def hess(x):
        return A

    return f, grad, hess, np.array([1.5, 1.0]), np.zeros(2)


def _rosenbrock():
    """Rosenbrock's banana-shaped valley."""
    return (optimize.rosen, optimize.rosen_der, optimize.rosen_hess,
            np.array([-1.2, 1.0]), np.ones(2))


CASES = [("Gaussian kernel", _gaussian), ("Badly-conditioned quadratic", _quadratic),
         ("Rosenbrock", _rosenbrock)]


def test_solver(optimizer):
    """Run ``optimizer(x0, f, f_grad, f_hessian)`` on the three test cases.

    The optimizer must return ``(all_x_k, all_f_k)``: the iterates and the
    objective values. For each case, plot the iterates over the level sets
    and ``f(x_k) - f(x*)`` on a log scale.
    """
    fig, axes = plt.subplots(2, len(CASES), figsize=(15, 8))
    for col, (name, make_case) in enumerate(CASES):
        print(f"\nRunning solver on case {col + 1}")
        f, f_grad, f_hessian, x0, x_star = make_case()
        all_x_k, all_f_k = optimizer(x0.copy(), f, f_grad, f_hessian)
        all_x_k = np.asarray(all_x_k)

        lo = np.minimum(all_x_k.min(axis=0), x_star) - 0.5
        hi = np.maximum(all_x_k.max(axis=0), x_star) + 0.5
        xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 200), np.linspace(lo[1], hi[1], 200))
        zz = np.array([f(np.array(p)) for p in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)

        ax = axes[0, col]
        ax.contour(xx, yy, zz, levels=30, cmap="viridis")
        ax.plot(all_x_k[:, 0], all_x_k[:, 1], "o-", color="tab:red", ms=3)
        ax.plot(*x_star, "k*", ms=12)
        ax.set_title(f"{name}: {len(all_x_k) - 1} iterations")

        # Recompute f on the iterates: a failed line search may return None values.
        f_values = np.array([f(x) for x in all_x_k])
        gap = np.maximum(f_values - f(x_star), 1e-16)
        axes[1, col].semilogy(gap)
        axes[1, col].set_xlabel("iteration")
        axes[1, col].set_ylabel("f(x_k) - f(x*)")
    plt.tight_layout()
    plt.show()
