from src.load_data import download_raw_data
from src.preprocess import clean, split_data
from src.train import train_logistic_regression, train_random_forest, train_gradient_boost
from src.inference import test_logistic_regression, test_random_forest, test_gradient_boost
from src.evaluate import evaluate_model

if __name__ == "__main__":
        # download dataset
        raw_data = download_raw_data()
        # clean the data
        clean_data = clean(raw_data)
        # split into test and training
        x_train, y_train, x_test, y_test = split_data(clean_data)

        # logistic regression model
        logistic_regression_model = train_logistic_regression(x_train, y_train, max_iter=1000, random_state=2025)
        y_pred, y_prob = test_logistic_regression(x_test)
        evaluate_model(y_pred=y_pred, y_test=y_test, y_prob=y_prob, model_name="log_reg")

        # random forest model
        random_forest_model = train_random_forest(x_train, y_train, n_estimators=500, random_state=2025)
        y_pred_rf, y_prob_rf = test_random_forest(x_test)
        evaluate_model(y_pred=y_pred_rf, y_test=y_test, y_prob=y_prob_rf, model_name="random_forest")

        # gradient boost model
        gradient_boost_model = train_gradient_boost(x_train, y_train, n_estimators=500, #300
                learning_rate=0.1,     
                max_depth=5,    
                subsample=1,  
                min_samples_split=10,
                min_samples_leaf=1,     
                max_features=None,
                random_state=2025)
        y_pred_gb, y_prob_gb = test_gradient_boost(x_test)
        evaluate_model(y_pred=y_pred_gb, y_test=y_test, y_prob=y_prob_gb, model_name="gradient_boost")