import joblib

def test_logistic_regression(x_test):
    scaler, model = joblib.load("models/logistic_regression_bundle.pkl")
    scaled_x_test = scaler.transform(x_test)
    y_pred = model.predict(scaled_x_test)
    return y_pred