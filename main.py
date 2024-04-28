import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm 
from sklearn.neighbors import NearestNeighbors

# Set page title and background color
st.set_page_config(page_title='Fashion Recommender System', page_icon=':shirt:', layout='wide', initial_sidebar_state='collapsed')

# Define custom CSS styles
custom_css = """
<style>
body {
    background-color: #f7f7f7; /* Light gray background */
    color: #333333; /* Dark text color */
    font-family: Arial, sans-serif; /* Font style */
}
.container {
    max-width: 1200px; /* Set maximum width */
    margin: auto; /* Center align content */
    padding: 20px; /* Add padding */
}
h1 {
    color: #1e88e5; /* Blue heading color */
}
</style>
"""

# Display custom CSS styles
st.markdown(custom_css, unsafe_allow_html=True)

# Main title
st.title('Fashion Recommender System')

# Load data and model
feature_list = np.array(pickle.load(open("embeddings.pkl","rb")))
file_names = pickle.load(open("filenames.pkl","rb"))
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists("uploads"):
            os.makedirs("uploads")
        with open(os.path.join("uploads", uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return False

# Function for feature extraction
def feature_extraction(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Function for recommendation
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload section
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.text("Your input Image:")
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", width=64)

        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Recommendation
        st.text("Recommended Products:")
        indices = recommend(features, feature_list)

        # Display recommendation
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(file_names[indices[0][0]])
        with col2:
            st.image(file_names[indices[0][1]])
        with col3:
            st.image(file_names[indices[0][2]])
        with col4:
            st.image(file_names[indices[0][3]])
        with col5:
            st.image(file_names[indices[0][4]])
    else:
        st.error("Some error occurred during file upload.")
