from src.load_data import download_raw_data
from src.preprocess import clean, split_data
from src.train import train_logistic_regression, train_random_forest
from src.inference import test_logistic_regression, test_random_forest
from src.evaluate import evaluate_model

if __name__ == "__main__":
        # download dataset
        raw_data = download_raw_data()
        # clean the data
        clean_data = clean(raw_data)
        # split into test and training
        x_train, y_train, x_test, y_test = split_data(clean_data)

        # logistic regression model
        logistic_regression_model = train_logistic_regression(x_train, y_train, max_iter=1000, seed=2025)
        y_pred, y_prob = test_logistic_regression(x_test)
        evaluate_model(y_pred=y_pred, y_test=y_test, y_prob=y_prob, model_name="log_reg")

        # random forest model
        random_forest_model = train_random_forest(x_train, y_train, n_estimators=100, seed=2025)
        y_pred_rf, y_prob_rf = test_random_forest(x_test)
        evaluate_model(y_pred=y_pred_rf, y_test=y_test, y_prob=y_prob_rf, model_name="random_forest")
