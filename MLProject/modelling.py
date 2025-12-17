import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import mlflow
import mlflow.sklearn

# 1. Enable Autologging
# DO NOT call mlflow.set_experiment() here if running via 'mlflow run'
mlflow.sklearn.autolog()

dataset_path = 'kc_house_data_preprocessing/'

# Load data
X_train = pd.read_csv(dataset_path+'X_train_preprocessed.csv', index_col=0)
X_test = pd.read_csv(dataset_path+'X_test_preprocessed.csv', index_col=0)
y_train = pd.read_csv(dataset_path+'train_preprocessed_with_target.csv', index_col=0).squeeze()
y_test = pd.read_csv(dataset_path+'test_preprocessed_with_target.csv', index_col=0).squeeze()

# 2. Use the existing run
# When started via 'mlflow run', this will automatically join the existing run
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Evaluation
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared: {r2:.4f}")
