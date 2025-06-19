import numpy as np
from scalpers_time_machine.model import train_model

def test_train_model_returns_outputs():
    model, X_test, y_test = train_model()

    assert model is not None, "Model should not be None"
    
    assert hasattr(X_test, 'shape'), "X_test should have shape attribute"
    assert hasattr(y_test, 'shape'), "y_test should have shape attribute"

def test_model_predict_output_shape():
    model, X_test, y_test = train_model()

    y_pred = model.predict(X_test)

    assert y_pred.shape == y_test.shape, "Predicted shape should match true labels"