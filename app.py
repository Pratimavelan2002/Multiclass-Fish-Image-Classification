# fish_classifier_app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pickle
from PIL import Image

# --- CONFIG ---
st.set_page_config(page_title="Fish Classifier", layout="wide")
st.title("🐟 Fish Image Classification App")

# --- MODEL SELECTION ---
st.sidebar.header("Select Model")

MODEL_FOLDER = "models"
ALL_MODELS = {
    "CNN": "cnn_model.h5",
    "VGG16": "vgg16_model.h5",
    "ResNet50": "resnet50_model.h5",
    "MobileNet": "mobilenet_model.h5",
    "InceptionV3": "inceptionv3_model.h5",
    "EfficientNetB0": "efficientnetb0_model.h5"
}

# Expected input sizes for each model
MODEL_INPUT_SIZES = {
    "CNN": (224, 224),
    "VGG16": (224, 224),
    "ResNet50": (224, 224),
    "MobileNet": (224, 224),
    "InceptionV3": (299, 299),
    "EfficientNetB0": (224, 224)
}

# Filter only existing models
model_options = {
    name: os.path.join(MODEL_FOLDER, fname)
    for name, fname in ALL_MODELS.items()
    if os.path.exists(os.path.join(MODEL_FOLDER, fname))
}

if not model_options:
    st.error("⚠️ No trained models found in the 'models/' folder.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Choose a model", list(model_options.keys()))
model_path = model_options[selected_model_name]

# --- LOAD MODEL ---
@st.cache_resource
def load_fish_model(path):
    if path.endswith(".h5"):
        return load_model(path)
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        return None

model = load_fish_model(model_path)

# --- LOAD CLASS LABELS ---
labels_path = model_path.replace(".h5", "_labels.txt").replace(".pkl", "_labels.txt")

if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        class_labels = [line.strip() for line in f.readlines()]
elif os.path.exists("class_labels.pkl"):
    with open("class_labels.pkl", "rb") as f:
        class_labels = pickle.load(f)
else:
    st.warning(f"No labels found for {selected_model_name}. Using generic labels.")
    class_labels = [f"Class {i}" for i in range(model.output_shape[-1])]

# --- IMAGE UPLOAD ---
st.sidebar.header("Upload Fish Image(s)")
uploaded_files = st.sidebar.file_uploader(
    "Choose one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    img_size = MODEL_INPUT_SIZES.get(selected_model_name, (224, 224))

    for uploaded_file in uploaded_files:
        st.subheader(f"Prediction for {uploaded_file.name}")
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize(img_size)

        img_array = np.array(img_resized) / 255.0
        img_array_expanded = np.expand_dims(img_array, axis=0)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        # --- PREDICTION ---
        predictions = model.predict(img_array_expanded)[0].flatten()

        if len(predictions) != len(class_labels):
            st.error(f"Model outputs ({len(predictions)}) do not match labels ({len(class_labels)}).")
        else:
            top_index = np.argmax(predictions)
            st.success(f"**Predicted Class:** {class_labels[top_index]}")
            st.info(f"**Confidence:** {predictions[top_index]*100:.2f}%")

            # Confidence for all classes
            conf_df = pd.DataFrame({"Class": class_labels, "Confidence": predictions * 100})
            st.subheader("Confidence Scores")
            st.bar_chart(conf_df.set_index("Class"))

# --- MODEL PERFORMANCE ---
st.sidebar.header("View Model Performance")
show_metrics = st.sidebar.checkbox("Show Training History / Confusion Matrix")

if show_metrics:
    st.subheader(f"Performance of {selected_model_name}")

    hist_path = f"reports/training_history_plots/{selected_model_name}_history.csv"
    if os.path.exists(hist_path):
        history = pd.read_csv(hist_path)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history['accuracy'], label='Train Accuracy')
        ax[0].plot(history['val_accuracy'], label='Val Accuracy')
        ax[0].set_title("Accuracy")
        ax[0].legend()
        ax[1].plot(history['loss'], label='Train Loss')
        ax[1].plot(history['val_loss'], label='Val Loss')
        ax[1].set_title("Loss")
        ax[1].legend()
        st.pyplot(fig)
    else:
        st.warning("Training history not found.")

    cm_path = f"reports/confusion_matrices/{selected_model_name}_confusion_matrix.csv"
    if os.path.exists(cm_path):
        cm = pd.read_csv(cm_path, index_col=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Confusion matrix not found.")

# --- COMPARISON REPORT ---
st.sidebar.header("Comparison Report")
show_comparison = st.sidebar.checkbox("Show All Models Comparison")

if show_comparison:
    comp_path = "reports/comparison_report.csv"
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        st.subheader("📊 Model Comparison Report")
        st.dataframe(comp_df)

        standard_metrics = ["Accuracy", "Precision", "Recall", "F1"]
        plot_cols = [c for c in standard_metrics if c in comp_df.columns]
        if not plot_cols:
            plot_cols = [c for c in comp_df.columns if c != "Model" and comp_df[c].dtype != "object"]

        if plot_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            comp_df.set_index("Model")[plot_cols].plot(kind="bar", ax=ax, colormap="Set2")
            plt.title("Model Performance Comparison")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("No numeric performance metrics found in comparison report to plot.")
    else:
        st.warning("Comparison report not found.")

# --- INDIVIDUAL CLASSIFICATION REPORTS (MERGED) ---
st.sidebar.header("Classification Reports")
show_class_reports = st.sidebar.checkbox("Show Merged Classification Reports")

if show_class_reports:
    report_files = {
        "CNN": "reports/CNN_classification_report.csv",
        "cnn_model": "reports/cnn_model_classification_report.csv"
    }

    merged_reports = []
    for model_name, path in report_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.insert(0, "Model", model_name)
            merged_reports.append(df)
        else:
            st.warning(f"Classification report not found for {model_name}.")

    if merged_reports:
        merged_df = pd.concat(merged_reports, ignore_index=True)
        st.subheader("📑 Combined Classification Reports")
        st.dataframe(merged_df)

        # Plot per-class F1 if available
        if "f1-score" in merged_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=merged_df, x="Model", y="f1-score", hue=merged_df.columns[1], ax=ax)
            plt.title("Per-class F1-Score Across Models")
            plt.ylabel("F1-Score")
            st.pyplot(fig)
