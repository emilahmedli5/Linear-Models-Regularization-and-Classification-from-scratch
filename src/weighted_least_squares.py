import numpy as np


class WeightedLeastSquares:

    def __init__(self) -> None:
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None) -> "WeightedLeastSquares":
        n = X.shape[0]

        # Append bias column
        X_aug = np.hstack((X, np.ones((n, 1))))

        # If no weights given, fall back to OLS (uniform weights)
        if sample_weights is None:
            sample_weights = np.ones(n)

        # Build the weighted normal equations without creating an n x n matrix
        # X^T W X = (X * w)^T X  and  X^T W y = (X * w)^T y
        XtW  = (X_aug * sample_weights[:, None]).T
        XtWX = XtW @ X_aug
        XtWy = XtW @ y

        self.weights, _, _, _ = np.linalg.lstsq(XtWX, XtWy, rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_aug = np.hstack((X, np.ones((X.shape[0], 1))))
        return X_aug @ self.weights