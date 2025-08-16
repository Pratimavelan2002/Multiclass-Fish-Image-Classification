import os, glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = r"C:\Users\prati\OneDrive\project4\data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
CONF_DIR = os.path.join(REPORTS_DIR, "confusion_matrices")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_dir = os.path.join(BASE_DIR, "test")
train_dir = os.path.join(BASE_DIR, "train")

datagen = ImageDataGenerator(rescale=1./255)
test_data = datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False
)
class_names = list(test_data.class_indices.keys())

def evaluate_model(model_path):
    model = tf.keras.models.load_model(model_path)
    y_true = test_data.classes
    y_probs = model.predict(test_data, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # ✅ Fix: align labels and target_names
    unique_labels = np.unique(y_true)
    target_names = [class_names[i] for i in unique_labels]

    report = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        target_names=target_names,
        output_dict=True
    )
    acc = report["accuracy"]

    # Save report
    name = os.path.splitext(os.path.basename(model_path))[0]
    pd.DataFrame(report).transpose().to_csv(
        os.path.join(REPORTS_DIR, f"{name}_classification_report.csv")
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, target_names, rotation=45, ha="right")
    plt.yticks(tick_marks, target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(os.path.join(CONF_DIR, f"{name}_cm.png"), bbox_inches="tight")
    plt.close(fig)

    return name, acc

rows = []
for path in glob.glob(os.path.join(MODELS_DIR, "*.h5")):
    name, acc = evaluate_model(path)
    rows.append({"Model": name, "Accuracy": acc})

df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
df.to_csv(os.path.join(REPORTS_DIR, "comparison_report.csv"), index=False)
print(df.to_string(index=False))
print("\nEvaluation complete. Reports saved in 'reports/'.")
