import numpy as np

class GaussianNaiveBayes:
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Estimates the mean, variance, and prior for each class.
        """
        self.classes_ = np.unique(y)
        self.parameters_ = {}
        self.priors_ = {}
        
        for c in self.classes_:
            # Filter data for the specific class
            X_c = X[y == c]
            
            # Calculate prior probability of the class P(y=c)
            self.priors_[c] = X_c.shape[0] / X.shape[0]
            
            # Estimate mean and variance for each feature j
            # Adding a tiny epsilon to variance to prevent division by zero during prediction
            self.parameters_[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0) + 1e-9 
            }
            
        return self

    def _calculate_log_pdf(self, class_idx, X):
        """
        Calculates the log of the Gaussian PDF for a given class.
        """
        mean = self.parameters_[class_idx]['mean']
        var = self.parameters_[class_idx]['var']
        
        # Log of Gaussian formula
        numerator = np.exp(-((X - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        
        # Add epsilon to prevent log(0)
        pdf = numerator / denominator
        return np.log(pdf + 1e-15)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class by maximizing the log-posterior.
        """
        # Store log-posteriors for each class
        log_posteriors = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, c in enumerate(self.classes_):
            prior = np.log(self.priors_[c])
            # Sum the log PDFs of all features (equivalent to product in regular space)
            log_likelihood = np.sum(self._calculate_log_pdf(c, X), axis=1)
            
            # log P(y=c|x) ∝ log(prior) + log(likelihood)
            log_posteriors[:, i] = prior + log_likelihood
            
        # Return the class with the highest log-posterior
        return self.classes_[np.argmax(log_posteriors, axis=1)]
    

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates class probabilities using the log-sum-exp trick for numerical stability.
        """
        log_posteriors = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, c in enumerate(self.classes_):
            prior = np.log(self.priors_[c])
            log_likelihood = np.sum(self._calculate_log_pdf(c, X), axis=1)
            log_posteriors[:, i] = prior + log_likelihood
            
        # Subtract max for numerical stability before exponentiating
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        exp_post = np.exp(log_posteriors - max_log)
        
        # Normalize to get probabilities
        return exp_post / np.sum(exp_post, axis=1, keepdims=True)


class MultinomialNaiveBayes:

    def __init__(self, alpha: float = 1.0) -> None:
        # alpha is the Laplace smoothing parameter to avoid zero probabilities
        self.alpha = alpha
        self.classes_ = None
        self.log_priors_ = {}
        self.log_likelihoods_ = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayes":
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        for c in self.classes_:
            X_c = X[y == c]

            # Log prior: log P(y = c)
            self.log_priors_[c] = np.log(X_c.shape[0] / n_samples)

            # Sum word counts across all documents in this class
            feature_counts = X_c.sum(axis=0)
            total_count = feature_counts.sum()

            # Laplace-smoothed log-likelihood: log P(word_j | class c)
            self.log_likelihoods_[c] = np.log(
                (feature_counts + self.alpha) / (total_count + self.alpha * n_features)
            )

        return self

    def _log_posterior(self, X: np.ndarray) -> np.ndarray:
        # log P(y=c | x) ∝ log P(y=c) + x · log P(x | y=c)
        log_posts = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            log_posts[:, i] = self.log_priors_[c] + X @ self.log_likelihoods_[c]
        return log_posts

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._log_posterior(X), axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_posts = self._log_posterior(X)
        # Subtract max before exponentiating to prevent overflow
        log_posts -= log_posts.max(axis=1, keepdims=True)
        probs = np.exp(log_posts)
        return probs / probs.sum(axis=1, keepdims=True)