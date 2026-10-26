
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="ChestScan AI",
    page_icon="🫁"
)

# GPU memory configuration
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# Load the fine-tuned model only once
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model(
        "/workspace/models/tl_model.keras",
        compile=False
    )

model = load_ai_model()


# Interface
st.title("🫁 ChestScan AI")
st.write("Pneumonia Detection Using Transfer Learning")

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Chest X-ray")

    if st.button("Predict"):

        # Preprocessing
        img = image.resize((180, 180))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = float(
            model(
                tf.convert_to_tensor(img_array),
                training=False
            ).numpy()[0][0]
        )

        # Results
        if prediction >= 0.5:
            st.warning("Prediction: PNEUMONIA")
        else:
            st.success("Prediction: NORMAL")

        st.write(f"Model score: {prediction:.2%}")

st.caption(
    "Educational demonstration only. "
    "Not intended for medical diagnosis."
)
