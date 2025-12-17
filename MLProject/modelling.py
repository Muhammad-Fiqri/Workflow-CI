# Import library
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import mlflow
import mlflow.sklearn

# mlflow.set_tracking_uri("http://127.0.0.1:5000") # Default local URI, Comment went running on Github Actions
mlflow.set_experiment("House_Price_Regression")

dataset_path = 'kc_house_data_preprocessing/'

# 1. Load the features (X) and target (y)
X_train = pd.read_csv(dataset_path+'X_train_preprocessed.csv', index_col=0)
X_test = pd.read_csv(dataset_path+'X_test_preprocessed.csv', index_col=0)
y_train = pd.read_csv(dataset_path+'train_preprocessed_with_target.csv', index_col=0)
y_test = pd.read_csv(dataset_path+'test_preprocessed_with_target.csv', index_col=0)

# 2. Verify the loading by checking shapes
print("Data Loading Successful!")
print(f"X_train shape: {X_train.shape}")
print(f"X_train head:\n{X_train.head()}")
print(f"X_test shape:  {X_test.shape}")
print(f"X_test head:\n{X_test.head()}")
print(f"y_train shape: {y_train.shape}")
print(f"y_train head:\n{y_train.head()}")
print(f"y_test shape:  {y_test.shape}")
print(f"y_test head:\n{y_test.head()}")

with mlflow.start_run(run_name="Linear_Regression_Base", nested=True):
    mlflow.log_param("model_type", "LinearRegression")
    
    # 3. Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 4. Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Manually logging custom metrics that autolog might not catch
    mlflow.log_metric("R-Squared", r2)
    mlflow.sklearn.log_model(model, artifact_path="house_price_model")

    print(f"R-squared: {r2:.4f}")  # Accuracy (closer to 1.0 is better)
    print(f"RMSE: ${rmse:.2f}")    # Average error in Dollars

    # The model is automatically saved as an artifact due to autolog()
    print("Model and metrics logged to MLflow successfully.")
