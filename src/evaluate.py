from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def evaluate_model(y_test, y_pred, y_prob, model_name="model"):
    # overall metrics
    print(classification_report(y_pred=y_pred, y_true=y_test))
    
    # text confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # visual confusion matrix
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.RdYlGn)
    plt.title("Confusion Matrix", pad=20)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], va='center', ha='center')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()
    fig.savefig(f"outputs/confusion_matrix_{model_name}.png")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob[:,1])

    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_score(y_test, y_prob[:,1]):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')  # random guess line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    fig.savefig(f"outputs/ROC_Curve_{model_name}.png")

    # Plot precision recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob[:,1])
    
    fig, ax = plt.subplots()
    plt.plot(recall, precision, label=f"AP = {average_precision_score(y_test, y_prob[:,1]):.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
    fig.savefig(f"outputs/precision_recall_curve_{model_name}.png")