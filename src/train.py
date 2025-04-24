import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# function to train a logistic regression model
def train_logistic_regression(x_train, y_train, **kwargs):
    from sklearn.linear_model import LogisticRegression

    pipe = build_pipeline(LogisticRegression(**kwargs), x_train, y_train)

    # save model
    save_model(pipe, "logistic_regression")

    return pipe

# function to train a logistic regression model
def train_random_forest(x_train, y_train, **kwargs):
    from sklearn.ensemble import RandomForestClassifier

    pipe = build_pipeline(RandomForestClassifier(**kwargs), x_train, y_train)

    # save model
    save_model(pipe, "random_forest")

    return pipe

def train_gradient_boost(x_train, y_train, **kwargs):
    from sklearn.ensemble import GradientBoostingClassifier

    pipe = build_pipeline(GradientBoostingClassifier(**kwargs), x_train, y_train)

    # save model
    save_model(pipe, "gradient_boost")

    return pipe

def build_pipeline(model, x_train, y_train, scaler=StandardScaler()):
    # Define pipeline
    pipe = Pipeline([
    ('scaler', scaler),   # define scaler
    ('classifier', model)      # define model
    ])
    
    # train pipeline
    pipe.fit(x_train, y_train)

    return pipe

def save_model(pipe, name):
    model_path = f"models/{name}_pipe.pkl"
    joblib.dump(pipe, model_path)

