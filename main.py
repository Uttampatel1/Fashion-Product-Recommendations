import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('models/embeddings.pkl','rb')))
filenames = pickle.load(open('models/filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

# Use 'st.set_page_config' to set the page title and icon
st.set_page_config(page_title="Fashion Recommender System", page_icon="👗")

# Use 'st.sidebar' for the sidebar content
st.sidebar.title("Upload an Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_column_width=True)
        
        # Feature extraction
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        
        # Recommendation
        indices = recommend(features, feature_list)
        
        # Show recommended images
        cols = st.columns(5)
        
        with cols[0]:
            st.image(filenames[indices[0][0]])
        with cols[1]:
            st.image(filenames[indices[0][1]])
        with cols[2]:
            st.image(filenames[indices[0][2]])
        with cols[3]:
            st.image(filenames[indices[0][3]])
        with cols[4]:
            st.image(filenames[indices[0][4]])
    else:
        st.sidebar.error("An error occurred during file upload")