import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

# Set up page configuration and title
st.set_page_config(page_title="KrishiNetra", page_icon="🌱", layout="centered")
st.title("KrishiNetra - Multi-Plant Disease Expert Classifier")

# 1. Dropdown selection for the user/farmer
plant_choice = st.selectbox(
    "Select the plant type:",
    ["Apple", "Grapes", "Potato","Corn", "Tomato"]
)

# 2. Configuration mapping for model paths and class labels
# ⚠️ CRITICAL: Ensure the order of labels matches 'train_generator.class_indices' exactly!
MODEL_MAP = {
    # "Apple": {
    #     "path": "04_models/model_apple_21.1.8.keras", 
    #     "labels": ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"]
    # },
    # "Grapes": {
    #     "path": "04_models/model_grapes_1.0.2.keras",
    #     "labels": ["Black Rot", "Esca (Black Measles)", "Leaf Blight", "Healthy"]
    # },
    "Potato": {
        "path": "potato_model_18.keras",
        "labels": ["Early Blight", "Late Blight", "Healthy"]
    },
    "Corn": {
    "path": "corn_model_11.keras",
    "labels": ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    },
#     "Tomato": {
#         "path": "04_models/model_tomato_1.0.1.keras", 
#         # Double check if 'Early Blight' should be added here based on your dataset split
#         "labels": ["Late Blight", "Leaf_Mold", "Septoria_leaf_spot", "Healthy"]
#     }
 }

# 3. Optimized Model Loading using Streamlit Caching
# This prevents reloading the heavy model from disk on every button click
@st.cache_resource
def load_expert_model(path):
    return tf.keras.models.load_model(path)

# 4. Image upload widget
uploaded_file = st.file_uploader(
    "Upload leaf image...",
    type=["jpg", "jpeg", "png", "jfif"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    use_crop = st.checkbox("Crop image before prediction")

    if use_crop:
        st.subheader("Select Leaf Region")

        image_to_predict = st_cropper(
            image,
            realtime_update=True,
            box_color="#00FF00",
            aspect_ratio=None,
            return_type="image"
        )

        st.image(image_to_predict, caption="Selected Region", use_container_width=True)

    else:
        image_to_predict = image

    if st.button("Predict Disease"): 
        with st.spinner("Loading expert model and analyzing image..."):
            
            # Retrieve specific model path and target labels based on user selection
            model_path = MODEL_MAP[plant_choice]["path"]
            current_labels = MODEL_MAP[plant_choice]["labels"]
            
            try:
                # Load the cached model instantly
                model = load_expert_model(model_path)
                
                # Preprocessing pipeline: Convert to RGB to drop Alpha/transparency channels
                image_rgb = image_to_predict.convert("RGB")
                
                # Resize image to match the target input dimensions expected by the CNN
                img = image_rgb.resize((224, 224))
                
                # Convert image to numpy array, normalize pixel values, and add batch dimension
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Execute inference/prediction
                predictions = model.predict(img_array)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                
                # Extract the corresponding disease class string label
                result = current_labels[predicted_class_idx]
                
                # 5. Output handling with confidence threshold intelligence
               # 5. Output handling with confidence threshold intelligence
                if confidence >= 70.0:
                    st.success(f"Prediction: **{result}** ({confidence:.2f}% confidence)")
                else:
                    st.warning(f"Prediction: **{result}** ({confidence:.2f}% confidence)")
                    st.info(
                        # "💡 **Note:**  \n"
                        "Low confidence may indicate mixed symptoms of more than one disease or background noise.  \n"
                        "For better results, upload a clear, well-lit, close-up photo of a single leaf."
                    )
                
            except OSError:
                st.error(f"❌ Model file not found! Please verify that the file exists at: `{model_path}`")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred during processing: {e}")