# do the same for input images

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors


# Load ResNet50 model with weights from ImageNet
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    
    GlobalMaxPooling2D()
])

feature_list = np.array(pickle.load(open("embeddings.pkl","rb")))
file_names = pickle.load(open("filenames.pkl","rb"))

img = image.load_img("sample/1163.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)    
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# using brute algorithm and euclidean distance to find nearest neighbor
neighbors  =NearestNeighbors(n_neighbors =5, algorithm='brute', metric='euclidean' )
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])
print(indices)

for file in indices[0]:
    print(file_names[file])