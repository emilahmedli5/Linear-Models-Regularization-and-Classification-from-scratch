import numpy as np
class RidgeRegression:
    def __init__(self, lambda_: float = 1.0)-> None:
        self.weights = None
        self.lambda_ = lambda_

    def fit(self, X: np.ndarray, y: np.ndarray)-> "RidgeRegression":
        X_aug = np.hstack((X, np.ones((X.shape[0], 1))))
        I_p = np.eye(X_aug.shape[1])
        I_p[-1, -1] = 0.0 
        self.weights = np.linalg.inv ((np.transpose(X_aug) @ X_aug) + (self.lambda_ * I_p))@np.transpose(X_aug) @ y

        return self


    def predict(self, X: np.ndarray)-> np.ndarray:
        augmented_X = np.hstack((X, np.ones((X.shape[0], 1))))
        
        return augmented_X @ self.weights