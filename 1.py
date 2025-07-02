# Import necessary libraries
import os  # For file system operations
from datetime import datetime  # To generate timestamp for saving files
import streamlit as st  # Streamlit for web app interface
import numpy as np  # For numerical operations
from tensorflow.keras.preprocessing import image  # For image preprocessing
from tensorflow.keras.applications import EfficientNetB7  # Pretrained EfficientNetB7 model
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions  # ImageNet utilities
from PIL import Image  # For image processing

# Streamlit app title
st.title("Image Classification with EfficientNetB7")

# Upload an image file from user (jpg, jpeg, png, webp)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

# If user uploads a file, then continue processing
if uploaded_file is not None:
    # Open the image using PIL and convert to RGB (in case of grayscale or RGBA)
    img = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image on the app
    st.image(img, caption='Uploaded Image', use_container_width=True)

    # Indicate classification process is beginning
    st.write("Classifying...")

    # Load the EfficientNetB7 model pre-trained on ImageNet dataset
    model = EfficientNetB7(weights='imagenet')

    # Resize the image to 600x600 pixels (required input size for EfficientNetB7)
    img_resized = img.resize((600, 600))

    # Convert image to array format for model prediction
    img_array = image.img_to_array(img_resized)

    # Add batch dimension since model expects batch input shape (1, 600, 600, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image as per EfficientNetB7 requirements (scaling, normalization, etc.)
    img_array = preprocess_input(img_array)

    # Make predictions using the model
    predictions = model.predict(img_array)

    # Decode predictions to human-readable labels (Top 3)
    results = decode_predictions(predictions, top=3)[0]

    # Display top-3 prediction results on Streamlit
    st.write("Top Predictions:")
    for i, (imagenetID, label, prob) in enumerate(results):
        st.write(f"{i+1}. {label}: {prob*100:.2f}%")

    # Button to save the image with timestamp
    if st.button("Save Image"):
        save_folder = "predicted_images"  # Folder to store images
        os.makedirs(save_folder, exist_ok=True)  # Create folder if it doesn't exist

        # Generate unique filename using current date and time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_folder, f"prediction_{timestamp}.png")

        # Save the image to the specified path
        img.save(save_path)

        # Show success message with save location
        st.success(f"Image saved to {save_path}")
