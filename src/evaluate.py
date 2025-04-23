from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(y_test, y_pred):
    print(classification_report(y_pred=y_pred, y_true=y_test))
    print(confusion_matrix(y_pred=y_pred, y_true=y_test))