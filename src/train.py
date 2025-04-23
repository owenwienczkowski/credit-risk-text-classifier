import joblib
from sklearn.preprocessing import StandardScaler

# function to train a logistic regression model
def train_logistic_regression(x_train, y_train):
    from sklearn.linear_model import LogisticRegression

    # define model
    model = LogisticRegression(max_iter=1000, random_state=2025)

    # scale model inputs
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    # train model
    model.fit(x_train_scaled, y_train)

    # save model
    model_path = "models/logistic_regression_bundle.pkl"
    joblib.dump((scaler, model), model_path)

    return model


