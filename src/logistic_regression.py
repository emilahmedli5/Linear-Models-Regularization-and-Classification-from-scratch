import numpy as np

def sigmoid(z):
    # Clip z to prevent overflow warnings in np.exp
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, lr: float = 0.1, lambda_: float = 0.0, max_iter: int = 1000) -> None:
        self.lr = lr
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.w_weights = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        row_num, col_num = X.shape
        
        # Include an intercept term by augmenting X with a column of ones
        X_aug = np.hstack((X, np.ones((row_num, 1))))
        
        # Initialize weights with col_num + 1 to account for the intercept
        self.w_weights = np.zeros(col_num + 1)
        self.loss_history = []

        for _ in range(self.max_iter):
            # 1. Calculate probabilities using the sigmoid link function
            z = np.dot(X_aug, self.w_weights)
            h = sigmoid(z)
            
            # 2. Calculate Objective Function J(w) (Negative Log-Likelihood + L2 Regularization)
            # Add a tiny epsilon to prevent log(0) mathematically
            eps = 1e-15 
            log_loss = -np.mean(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))
            
            # Regularize weights (excluding the intercept/bias which is the last element)
            l2_reg = (self.lambda_ / (2 * row_num)) * np.sum(self.w_weights[:-1]**2)
            self.loss_history.append(log_loss + l2_reg)
            
            # 3. Calculate gradients
            error = h - y
            grad_w = (1 / row_num) * np.dot(X_aug.T, error)
            
            # Gradient of the regularization term
            grad_reg = (self.lambda_ / row_num) * self.w_weights
            grad_reg[-1] = 0.0 # Do not penalize the intercept
            
            # 4. Update weights
            self.w_weights -= self.lr * (grad_w + grad_reg)

        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        row_num = X.shape[0]
        # Augment X to account for the intercept
        X_aug = np.hstack((X, np.ones((row_num, 1))))
        
        # Return the raw probabilities
        return sigmoid(np.dot(X_aug, self.w_weights))

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Threshold the probabilities at 0.5 for binary classification
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)
    


class MulticlassLogisticRegressionOvR:
    def __init__(self, lr: float = 0.1, lambda_: float = 0.0, max_iter: int = 1000):
        """
        Initializes the OvR multiclass logistic regression model.
        """
        self.lr = lr
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.models = []
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MulticlassLogisticRegressionOvR":
        """
        Trains K binary classifiers, one for each class.
        """
        self.classes_ = np.unique(y)
        self.models = []
        
        for c in self.classes_:
            # Create binary targets: 1 if the class is 'c', 0 otherwise
            y_binary = np.where(y == c, 1, 0)
            
            # Initialize a new binary LogisticRegression model 
            # (Assuming the LogisticRegression class from the previous step is available)
            model = LogisticRegression(lr=self.lr, lambda_=self.lambda_, max_iter=self.max_iter)
            
            # Train the binary model
            model.fit(X, y_binary)
            
            # Store the trained model
            self.models.append(model)
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts class labels by selecting the class with the highest probability.
        """
        # Array to store probabilities from each model: shape (n_samples, n_classes)
        probs = np.zeros((X.shape[0], len(self.classes_)))
        
        for idx, model in enumerate(self.models):
            # Get the probability of the positive class (class 'c')
            probs[:, idx] = model.predict_proba(X)
            
        # Find the index of the model with the maximum probability for each sample
        best_class_indices = np.argmax(probs, axis=1)
        
        # Map indices back to the actual class labels
        return self.classes_[best_class_indices]