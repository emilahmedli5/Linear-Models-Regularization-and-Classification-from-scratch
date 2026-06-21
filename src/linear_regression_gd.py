import numpy as np

class LinearRegressionGD:
    """
    Linear Regression model optimized using Batch Gradient Descent.
    """
    def __init__(self, lr: float = 0.01, max_iter: int = 1000,
                 tol: float = 1e-6) -> None: 
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        """Fit the linear regression model using gradient descent."""
        row_num, col_num = X.shape
        X_aug = np.hstack((X, np.ones((row_num, 1))))
        
        # Initialize weights with col_num + 1 to account for the intercept
        self.w_weights = np.zeros(col_num + 1)
        
        # Initialize as an empty list to track the scalar MSE
        self.loss_history = []

        # Iterate over a range, not the integer itself
        for i in range(self.max_iter):
            y_pred = np.dot(X_aug, self.w_weights)
            e = y - y_pred
            
            # Calculate and append the scalar Mean Squared Error
            mse = np.mean(e**2)
            self.loss_history.append(mse)
            
            grad_w = (-2/row_num) * (np.transpose(X_aug) @ e)

            # Store the old weights before updating to check for convergence
            old_weights = self.w_weights.copy()

            # Corrected variable name typo
            self.w_weights -= self.lr * grad_w

            # Check if the maximum weight change is below the tolerance threshold
            if np.max(np.abs(self.w_weights - old_weights)) < self.tol:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for the given input features."""
        row_num, col_num = X.shape
        X_aug = np.hstack((X, np.ones((row_num, 1))))  

        # Corrected matrix multiplication order
        return X_aug @ self.w_weights