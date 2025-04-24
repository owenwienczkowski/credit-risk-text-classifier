import joblib

def test_logistic_regression(x_test):
    pipe = joblib.load("models/logistic_regression_pipe.pkl")
    scaled_x_test = pipe.transform(x_test)
    y_pred = pipe.predict(scaled_x_test)
    y_prob = pipe.predict_proba(scaled_x_test)
    return y_pred, y_prob