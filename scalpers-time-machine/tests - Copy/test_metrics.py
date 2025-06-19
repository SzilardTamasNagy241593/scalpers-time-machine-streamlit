import numpy as np
from scalpers_time_machine.metrics import evaluate_model

class DummyModel:
    def predict(self, X):
        return X  

def test_evaluate_model_returns_expected_keys():
    X_test = np.array([0.1, 0.2, 0.3, 0.4])
    y_test = np.array([0.1, 0.2, 0.3, 0.4])
    model = DummyModel()

    result = evaluate_model(model, X_test, y_test)

    expected_keys = ["mae", "rmse", "r2", "accuracy", "precision", "recall", "f1_score"]
    for key in expected_keys:
        assert key in result, f"{key} should be in the evaluation result"

def test_evaluate_model_metric_values():
    X_test = np.array([0.1, 0.2, 0.3, 0.4])
    y_test = np.array([0.1, 0.2, 0.3, 0.4])
    model = DummyModel()

    result = evaluate_model(model, X_test, y_test)

    assert result["mae"] == 0
    assert result["rmse"] == 0
    assert result["r2"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0