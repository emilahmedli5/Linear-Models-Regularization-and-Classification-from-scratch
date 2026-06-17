import numpy as np

class LinearRegression:


    def __init__(self)-> None:
        self.weights = None


    def fit(self, X: np.ndarray, y: np.ndarray)-> "LinearRegression":
        X_aug = np.hstack((X, np.ones((X.shape[0], 1))))
        A=np.transpose(X_aug)@X_aug
        b=np.transpose(X_aug)@y

        self.weights=np.linalg.solve(A,b)


        return self

    def predict(self, X: np.ndarray)-> np.ndarray:
        
        augmented_X = np.hstack((X, np.ones((X.shape[0], 1))))

        return augmented_X@self.weights
