from src.load_data import download_raw_data
from src.preprocess import clean, split_data
from src.train import train_logistic_regression
from src.inference import test_logistic_regression
from src.evaluate import evaluate_model

if __name__ == "__main__":
        raw_data = download_raw_data()
        clean_data = clean(raw_data)
        x_train, y_train, x_test, y_test = split_data(clean_data)
        logistic_regression_model = train_logistic_regression(x_train, y_train)
        y_pred = test_logistic_regression(x_test)
        evaluate_model(y_pred=y_pred, y_test=y_test)