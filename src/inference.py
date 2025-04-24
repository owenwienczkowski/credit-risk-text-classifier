import joblib

def test_logistic_regression(x_test):
    return test_model("models/logistic_regression_pipe.pkl", x_test)

def test_random_forest(x_test):
    return test_model("models/random_forest_pipe.pkl", x_test)

def test_model(path, x_test):
    pipe = joblib.load(path)
    y_pred = pipe.predict(x_test)
    y_prob = pipe.predict_proba(x_test)
    return y_pred, y_prob