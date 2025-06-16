# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import utils as ut
#
#
# def run(model, log_dir):
#     print("evaluating")
#
#     val_generator = ut.load_data(ut.DATA_VAL_PATH, shuffle=False)
#     y_true = val_generator.classes
#     y_pred_probs = model.predict(val_generator)
#     y_pred = np.argmax(y_pred_probs, axis=1)
#
#     cm = confusion_matrix(y_true, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(val_generator.class_indices.keys()))
#     disp.plot(cmap=plt.cm.Blues)
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#
#     fig = plt.gcf()
#     fig.savefig(f"{log_dir}/confusion_matrix.png")
#     print(f"Confusion matrix saved to {log_dir}/confusion_matrix.png")
#
#     plt.show()
#

import utils as ut
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def run(model, log_dir):
    print("evaluating")

    val_generator = ut.load_data(ut.DATA_VAL_PATH, shuffle=False)
    y_true = val_generator.classes

    # predict
    y_pred_probs = model.predict(val_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"Validation Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(val_generator.class_indices.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # save
    fig = plt.gcf()
    fig.savefig(f"{log_dir}/confusion_matrix.png")
    print(f"Confusion matrix saved to {log_dir}/confusion_matrix.png")

    # show
    plt.show()

    return acc
