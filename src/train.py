import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# function to train a logistic regression model
def train_logistic_regression(x_train, y_train):
    from sklearn.linear_model import LogisticRegression

    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=2025), x_train, y_train)

    # save model
    model_path = "models/logistic_regression_pipe.pkl"
    joblib.dump(pipe, model_path)

    return 

# function to train a logistic regression model
def train_random_forest(x_train, y_train):
    from sklearn.ensemble import RandomForestClassifier

    pipe = build_pipeline(RandomForestClassifier(n_estimators=100, random_state=2025), x_train, y_train)

    # save model
    model_path = "models/random_forest_pipe.pkl"
    joblib.dump(pipe, model_path)

    return 



def build_pipeline(model, x_train, y_train):
    # Define pipeline
    pipe = Pipeline([
    ('scaler', StandardScaler()),   # define scaler
    ('classifier', model)      # define model
    ])
    
    # train pipeline
    pipe.fit(x_train, y_train)

    return pipe