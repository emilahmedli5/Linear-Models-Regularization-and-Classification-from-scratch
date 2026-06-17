import pytest
import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLR

# Import your custom class from your file
from src.linear_regression import LinearRegression as CustomLR

def test_predictions_match_sklearn():
    """
    Test that the custom OLS implementation matches scikit-learn
    within a tolerance of 10^-9.
    """
    # 1. Generate a random test matrix and target vector
    np.random.seed(42)
    n_samples, n_features = 100, 5
    
    X_test = np.random.rand(n_samples, n_features)
    true_weights = np.random.rand(n_features)
    y_test = X_test @ true_weights + (np.random.rand(n_samples) * 0.5)

    # 2. Train and predict using YOUR custom model
    my_model = CustomLR()
    my_preds = my_model.fit(X_test, y_test).predict(X_test)

    # 3. Train and predict using Scikit-Learn's model
    sk_model = SklearnLR()
    sk_preds = sk_model.fit(X_test, y_test).predict(X_test)

    # 4. Assert that the outputs match within the strict 1e-9 tolerance
    # If the difference exceeds this, pytest will automatically fail the test and print why
    np.testing.assert_allclose(my_preds, sk_preds, atol=1e-9, rtol=0)

def test_fit_returns_self():
    """
    Verify that the fit method returns the instance itself to support method chaining.
    """
    np.random.seed(42)
    X = np.random.rand(10, 2)
    y = np.random.rand(10)
    
    model = CustomLR()
    returned_object = model.fit(X, y)
    
    assert returned_object is model, "The fit method must return 'self'"