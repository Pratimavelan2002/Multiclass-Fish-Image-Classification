import os, json, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0 # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

# USER PATHS 

BASE_DIR = r"C:\Users\prati\OneDrive\project4\data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
CONF_DIR = os.path.join(REPORTS_DIR, "confusion_matrices")
PLOTS_DIR = os.path.join(REPORTS_DIR, "training_history_plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CONF_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_CNN = 10
EPOCHS_TL = 5

train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")
test_dir = os.path.join(BASE_DIR, "test")


# Data generators

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
val_data = val_test_datagen.flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
test_data = val_test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

num_classes = len(train_data.class_indices)
class_names = list(train_data.class_indices.keys())

def plot_history(history, title, out_path):
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.savefig(out_path.replace(".png", "_acc.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.savefig(out_path.replace(".png", "_loss.png"), bbox_inches="tight")
    plt.close()

def evaluate_and_save(model, name):
    y_true = test_data.classes
    y_probs = model.predict(test_data, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_csv = os.path.join(REPORTS_DIR, f"{name}_classification_report.csv")
    report_df.to_csv(report_csv, index=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(os.path.join(CONF_DIR, f"{name}_cm.png"), bbox_inches="tight")
    plt.close(fig)

    # Return accuracy
    acc = report["accuracy"]
    return acc


#  CNN from scratch

cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])
cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
hist_cnn = cnn.fit(train_data, validation_data=val_data, epochs=EPOCHS_CNN, verbose=1)
cnn.save(os.path.join(MODELS_DIR, "cnn_model.h5"))
plot_history(hist_cnn, "CNN", os.path.join(PLOTS_DIR, "CNN.png"))
acc_cnn = evaluate_and_save(cnn, "CNN")


#  Transfer Learning models

bases = {
    "VGG16": VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    "ResNet50": ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    "MobileNet": MobileNet(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    "InceptionV3": InceptionV3(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    "EfficientNetB0": EfficientNetB0(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
}

results = {"Model": [], "Accuracy": []}

for name, base in bases.items():
    print(f"\nTraining {name}...")
    base.trainable = False  # feature extractor
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(train_data, validation_data=val_data, epochs=EPOCHS_TL, verbose=1)
    model.save(os.path.join(MODELS_DIR, f"{name}_model.h5"))
    plot_history(hist, name, os.path.join(PLOTS_DIR, f"{name}.png"))
    acc = evaluate_and_save(model, name)
    results["Model"].append(name)
    results["Accuracy"].append(acc)


# Comparison report
results["Model"].insert(0, "CNN")
results["Accuracy"].insert(0, acc_cnn)
df = pd.DataFrame(results)
df.to_csv(os.path.join("reports", "comparison_report.csv"), index=False)
print("\n=== Comparison ===")
print(df.sort_values("Accuracy", ascending=False).to_string(index=False))

print("\nTraining complete. Models saved in 'models/'. Reports in 'reports/'.")
