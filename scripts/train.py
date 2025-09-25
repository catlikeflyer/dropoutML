import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.lightgbm

# Set up MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Student Dropout Prediction")

# Load processed data
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

with mlflow.start_run():
    # Define model parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'n_estimators': 100,
        'learning_rate': 0.05,
        'num_leaves': 20,
        'max_depth': 5,
        'scale_pos_weight': (len(y_train) - sum(y_train)) / sum(y_train) # Handles class imbalance
    }
    mlflow.log_params(params)

    # Train the model
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log the model artifact
    mlflow.lightgbm.log_model(model, "model")
    print(f"Model logged to MLflow with run_id: {mlflow.active_run().info.run_id}")