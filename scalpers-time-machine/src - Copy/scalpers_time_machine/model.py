import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from scalpers_time_machine.data_loader import load_data
from scalpers_time_machine.config import FEATURE_COLUMNS, TARGET_COLUMN

logging.basicConfig(level=logging.INFO)


def train_model():
    df = load_data()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Test MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.4f}")

    # Classification metrics (return >= 0 => 1, else 0)
    y_test_class = (y_test >= 0).astype(int)
    y_pred_class = (y_pred >= 0).astype(int)

    accuracy = accuracy_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class)
    recall = recall_score(y_test_class, y_pred_class)
    f1 = f1_score(y_test_class, y_pred_class)
    logging.info(f"Classification Accuracy: {accuracy:.4f}")
    logging.info(f"Classification Precision: {precision:.4f}")
    logging.info(f"Classification Recall: {recall:.4f}")
    logging.info(f"Classification F1: {f1:.4f}")

    return model, X_test, y_test


if __name__ == "__main__":
    model, X_test, y_test = train_model()
    print("Model trained and ready!")