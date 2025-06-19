import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from mlflow.models import infer_signature


# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Predict Income")


if __name__ == "__main__":
    train_data = pd.read_csv('sgdata_preprocessing/sgtrain.csv')
    test_data = pd.read_csv('sgdata_preprocessing/sgtest.csv')

    train_data = train_data.dropna()
    test_data = test_data.dropna()

    X_train = train_data.drop("Income", axis=1)
    y_train = train_data["Income"]

    X_test = test_data.drop("Income", axis=1)
    y_test = test_data["Income"]
    input_example = X_train[0:5]

    mlflow.sklearn.autolog()
    model = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    signature = infer_signature(X_train, predictions)
    
    # menyimpan model ke artifact
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="basic_model",
        input_example=input_example,
        signature=signature
    )
