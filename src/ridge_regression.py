import numpy as np
class RidgeRegression:
    def __init__(self, lambda_: float = 1.0)-> None:
        self.weights = None
        self.lambda_ = lambda_

    def fit(self, X: np.ndarray, y: np.ndarray)-> "RidgeRegression":
        X_aug = np.hstack((X, np.ones((X.shape[0], 1))))
        
        # Create the Identity Matrix
        # The size is the number of columns in our augmented matrix
        I_p = np.eye(X_aug.shape[1])
        
        # CRITICAL: We do NOT want to regularize the intercept. 
        # Since our intercept is the last column, we set the last diagonal element to 0.
        I_p[-1, -1] = 0.0 
        
        # Set up the normal equations: (X^T X + lambda * I_p)w = X^T y
        A = (np.transpose(X_aug) @ X_aug) + (self.lambda_ * I_p)
        b = np.transpose(X_aug) @ y

        # Solve the linear system for numerical stability
        self.weights = np.linalg.solve(A, b)

        return self


    def predict(self, X: np.ndarray)-> np.ndarray:
        augmented_X = np.hstack((X, np.ones((X.shape[0], 1))))
        
        # Return the dot product
        return augmented_X @ self.weights