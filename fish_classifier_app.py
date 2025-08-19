
import streamlit as st
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0
)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import numpy as np
from PIL import Image

st.title("🐟 Fish Classifier App")
st.write("Upload a fish image and select a pre-trained model to predict the category.")

# Model selection
model_name = st.selectbox(
    "Choose a model",
    ("VGG16", "ResNet50", "MobileNet", "InceptionV3", "EfficientNetB0")
)

#  Load the selected model
@st.cache_resource(show_spinner=True)
def load_model(name):
    if name == "VGG16":
        model = VGG16(weights="imagenet")
        preprocess = vgg_preprocess
        target_size = (224, 224)
    elif name == "ResNet50":
        model = ResNet50(weights="imagenet")
        preprocess = resnet_preprocess
        target_size = (224, 224)
    elif name == "MobileNet":
        model = MobileNet(weights="imagenet")
        preprocess = mobilenet_preprocess
        target_size = (224, 224)
    elif name == "InceptionV3":
        model = InceptionV3(weights="imagenet")
        preprocess = inception_preprocess
        target_size = (299, 299)
    elif name == "EfficientNetB0":
        model = EfficientNetB0(weights="imagenet")
        preprocess = efficientnet_preprocess
        target_size = (224, 224)
    return model, preprocess, target_size

model, preprocess_input, target_size = load_model(model_name)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image 
    img_resized = img.resize(target_size)
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # --- Predict ---
    preds = model.predict(x)
    
    # Decode predictions 
    from tensorflow.keras.applications.imagenet_utils import decode_predictions
    decoded = decode_predictions(preds, top=3)[0]

    st.write("### Top 3 Predictions:")
    for i, (imagenet_id, label, prob) in enumerate(decoded):
        st.write(f"{i+1}. {label}: {prob*100:.2f}%")
