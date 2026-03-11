# ML Optimization — From Scratch Implementations

A collection of Jupyter notebooks implementing and benchmarking core optimization algorithms used in machine learning, from first-order gradient methods to second-order quasi-Newton approaches. Each notebook derives the theory, implements the algorithm from scratch (NumPy/Python), and produces convergence analysis.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Numba](https://img.shields.io/badge/Numba-00A3E0?logo=numba&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

---

## Skills Demonstrated

| Area | Topics |
|---|---|
| **Convex Optimization** | Gradient descent, proximal operators, Nesterov acceleration, convergence proofs |
| **Stochastic Methods** | SGD, SAG, SVRG, variance reduction, step-size tuning |
| **Second-Order Methods** | Newton's method, BFGS, L-BFGS, Hessian approximation |
| **Regularization** | L1 (Lasso), L2 (Ridge), L2,1 (Group Lasso), soft-thresholding |
| **Linear Algebra** | Lipschitz constants, condition numbers, positive-definite matrices, Toeplitz covariance |
| **Classification** | Logistic regression, multinomial softmax, digits recognition |
| **Software** | NumPy, scikit-learn API, Numba JIT, matplotlib, convergence benchmarking |

---

## Notebooks

### [gradient_descent_regression.ipynb](gradient_descent_regression.ipynb)
**Proximal Gradient Descent & Nesterov Acceleration**

Implements PGD and APGD (accelerated) for linear and logistic regression with L1/L2 regularization. Derives proximal operators analytically (soft-thresholding for L1, shrinkage for L2) and validates gradients numerically. Studies the impact of feature correlation and regularization strength on convergence speed.

- APGD converges 2–3× faster than PGD (O(1/k²) vs O(1/k))
- Higher feature correlation degrades convergence — quantified with condition numbers
- Regularization trade-off: larger λ → faster convergence but higher bias

---

### [stichastic_gradient_descent.ipynb](stichastic_gradient_descent.ipynb)
**SGD, SAG, and SVRG — Variance Reduction Methods**

Benchmarks 5 optimizers (GD, AGD, SGD, SAG, SVRG) on large-scale ridge regression (10 000 samples). Implements SAG's gradient memory table and SVRG's inner/outer loop from scratch with Numba JIT for performance. Analyzes convergence under varying correlation (ρ = 0.1 → 0.9) and regularization (λ = 0.0001 → 1).

- SVRG achieves ~10⁻⁸ precision; SGD saturates due to gradient noise
- SAG/SVRG robust to high feature correlation where batch GD degrades
- Derives linear convergence rates O((n + L_max/μ) log(1/ε)) for variance-reduced methods

---

### [coordinate_gradient_descent.ipynb](coordinate_gradient_descent.ipynb)
**Coordinate Descent — Cyclic, Greedy, and Proximal**

Implements and compares cyclic CD vs greedy CD for OLS, then proximal coordinate descent for sparse logistic regression. Benchmarks against ISTA on the Leukemia dataset (7 129 features, 72 samples) — a high-dimensional setting where CD naturally scales.

- Proves λ_max threshold: above it, the sparse solution is identically zero
- Greedy CD faster than cyclic on correlated features
- Proximal CD reaches 10⁻⁴ precision on leukemia data vs 10⁻³ for ISTA

---

### [quasi_newton_methods.ipynb](quasi_newton_methods.ipynb)
**Newton, DFP, BFGS, and L-BFGS**

Full from-scratch implementation of Newton's method (with eigenvalue regularization), DFP, BFGS, and L-BFGS (two-loop recursion). All methods use line search with strong Wolfe conditions. Tested on three functions: Gaussian kernel (non-convex), badly-conditioned quadratic, and Rosenbrock.

- BFGS: best trade-off — superlinear convergence without computing the Hessian
- L-BFGS: matches BFGS accuracy with O(nm) memory instead of O(n²)
- Newton: quadratic convergence near the optimum, but requires PD Hessian regularization

---

### [multinomial_logistic_nonsmooth_optimization.ipynb](multinomial_logistic_nonsmooth_optimization.ipynb)
**Multinomial Logistic Regression with L1 / L2 / L2,1 Regularization**

Builds a scikit-learn compatible estimator for multi-class classification with three regularizers. Proves convexity for each penalty, derives the corresponding proximal operators, and evaluates on the digits dataset (10 classes). Implements L2,1 (group Lasso) which zeros out entire feature groups — useful for structured feature selection.

- L2,1 norm induces row-sparsity across all classes simultaneously
- All three regularizers yield convex problems — proved from first principles
- Clean sklearn API: `fit / predict / score` with cross-validation ready

---

## Setup

```bash
pip install numpy scipy scikit-learn matplotlib numba
jupyter notebook
```

---

## Context

These notebooks were produced as part of a graduate-level optimization for machine learning course. The focus is on understanding *why* algorithms converge, not just running them — every method includes derivation, implementation, and empirical validation.
