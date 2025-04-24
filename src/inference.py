import joblib

def test_logistic_regression(x_test):
    pipe = joblib.load("models/logistic_regression_pipe.pkl")
    y_pred = pipe.predict(x_test)
    y_prob = pipe.predict_proba(x_test)
    return y_pred, y_prob