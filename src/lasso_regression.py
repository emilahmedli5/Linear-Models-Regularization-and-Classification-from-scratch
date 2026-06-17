import numpy as np

class LassoRegression:
    def __init__(self, lambda_: float = 1.0, max_iter: int = 10000, tol: float = 1e-4) -> None:
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        
        # Variables to store standardization parameters
        self.X_mean = None
        self.X_std = None
        self.y_mean = None

    def _soft_threshold(self, z: float, lambda_: float) -> float:
        """The soft-thresholding operator S(z, lambda)"""
        return np.sign(z) * np.maximum(np.abs(z) - lambda_, 0.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegression":
        n_samples, n_features = X.shape
        
        # 1. Standardize Features (zero mean, unit variance)
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        # Prevent division by zero for constant features
        self.X_std[self.X_std == 0] = 1.0 
        X_scaled = (X - self.X_mean) / self.X_std
        
        # 2. Center the target variable
        self.y_mean = np.mean(y)
        y_centered = y - self.y_mean
        
        # Initialize weights to zero
        self.weights = np.zeros(n_features)
        
        # Precompute the denominator: sum of squared feature values
        # Since X is standardized, this is mathematically equivalent to the variance * n_samples
        Z = np.sum(X_scaled**2, axis=0)
        
        # 3. Cyclic Coordinate Descent
        for _ in range(self.max_iter):
            w_old = self.weights.copy()
            
            for j in range(n_features):
                # Current predictions using all features
                y_pred = X_scaled @ self.weights
                
                # Calculate the residual excluding feature j, fully vectorized over samples (i)
                # rho_j represents the numerator: sum( x_ij * (y_i - y_pred_without_j) )
                rho_j = np.sum(X_scaled[:, j] * (y_centered - y_pred + (X_scaled[:, j] * self.weights[j])))
                
                # Update weight j using the soft-thresholding operator
                self.weights[j] = self._soft_threshold(rho_j, self.lambda_) / Z[j]
                
            # Convergence check: stop if the largest weight update is smaller than tolerance
            if np.max(np.abs(self.weights - w_old)) < self.tol:
                break
                
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Standardize the incoming test data using the parameters learned during training
        X_scaled = (X - self.X_mean) / self.X_std
        
        # Generate predictions and add the y_mean back in to act as the intercept
        return (X_scaled @ self.weights) + self.y_mean