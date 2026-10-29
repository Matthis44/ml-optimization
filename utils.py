"""Plotting helper for the multinomial logistic regression notebook.

Re-implementation of the helper provided with the course.
"""

import matplotlib.pyplot as plt
import numpy as np


def show_decision_boundary(predict_proba, data, n_points=300, ax=None):
    """Colour the plane by the predicted class of a 2-D, K-class classifier.

    Parameters
    ----------
    predict_proba : callable
        Takes an array of shape (n, 2) and returns probabilities (n, K).
    data : tuple (X, Y)
        Samples of shape (n, 2) and one-hot targets (n, K) or labels (n,).
    """
    X, Y = data
    labels = np.argmax(Y, axis=1) if np.ndim(Y) == 2 else np.asarray(Y)
    margin = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - margin, X[:, 0].max() + margin, n_points),
        np.linspace(X[:, 1].min() - margin, X[:, 1].max() + margin, n_points),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    predicted = np.argmax(predict_proba(X=grid), axis=1).reshape(xx.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    n_classes = max(labels.max(), predicted.max()) + 1
    ax.contourf(xx, yy, predicted, levels=np.arange(n_classes + 1) - 0.5, cmap="viridis", alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", edgecolor="k", s=15)
    ax.set_title("Decision boundary")
    plt.show()
    return ax
