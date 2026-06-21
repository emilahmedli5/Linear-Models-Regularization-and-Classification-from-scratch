import numpy as np


class WeightedLeastSquares:
    """
    Weighted Least Squares (WLS) regression.

    Solves the weighted normal equations:
        w* = (X^T W X)^{-1} X^T W y
    where W = diag(sample_weights) is a diagonal matrix of per-sample weights.

    When all sample weights are equal this reduces exactly to OLS.
    """

    def __init__(self) -> None:
        self.weights = None  # learned parameter vector [w_1,...,w_p, bias]

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None
    ) -> "WeightedLeastSquares":
        """
        Fit WLS.

        Parameters
        ----------
        X              : (n, p) feature matrix
        y              : (n,)   target vector
        sample_weights : (n,)   non-negative per-sample importance weights.
                                Defaults to uniform (= OLS).
        """
        n = X.shape[0]

        # Augment X with a bias column of ones
        X_aug = np.hstack((X, np.ones((n, 1))))  # (n, p+1)

        # Default to uniform weights → OLS
        if sample_weights is None:
            sample_weights = np.ones(n)

        sample_weights = np.asarray(sample_weights, dtype=float)
        if sample_weights.shape != (n,):
            raise ValueError(
                f"sample_weights must have shape ({n},), got {sample_weights.shape}"
            )
        if np.any(sample_weights < 0):
            raise ValueError("sample_weights must be non-negative.")

        # Build diagonal weight matrix efficiently: W = diag(sample_weights)
        # X^T W X  = (X * w[:, None])^T X   — avoids materialising the n×n W matrix
        XtW  = (X_aug * sample_weights[:, None]).T   # (p+1, n)
        XtWX = XtW @ X_aug                            # (p+1, p+1)
        XtWy = XtW @ y                                # (p+1,)

        # Solve normal equations using lstsq for numerical stability
        self.weights, _, _, _ = np.linalg.lstsq(XtWX, XtWy, rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_aug = np.hstack((X, np.ones((X.shape[0], 1))))
        return X_aug @ self.weights