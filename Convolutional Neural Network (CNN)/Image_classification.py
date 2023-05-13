# Image classification using pre-trained CNN model efficientnet
from tensorflow import keras
# preprocess_input is a pass-through function for EffNets
from keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

#urllib is a package that collects several modules for working with URLs: urllib. request is for opening and reading URLs.
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

# Public domain image
#url = 'https://upload.wikimedia.org/wikipedia/commons/0/02/Black_bear_large.jpg'
#url='https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Grosser_Panda.JPG/1920px-Grosser_Panda.JPG'
#url='https://upload.wikimedia.org/wikipedia/commons/d/d2/Petronas_Panorama_II_%284to3%29.jpg'
url='https://thumbs.dreamstime.com/z/wide-selection-toys-children-s-store-inside-toy-shop-rows-shelves-joy-135605121.jpg'
#PIL is a popular image processing library. Here it is used to load images as NumPy arrays
import PIL
img = PIL.Image.open(urllib.request.urlopen(url))
img = img.resize((224, 224))
img_batch = np.expand_dims(img, 0)

# Load model and run prediction
effnet = keras.applications.EfficientNetV2B0(weights='imagenet', include_top=True)
pred = effnet.predict(img_batch)
print(decode_predictions(pred))# prints class number, class name, and confidence of prediction
plt.imshow(img)
plt.title(f'Class: {decode_predictions(pred)[0][0][1]}\nConfidence: {decode_predictions(pred)[0][0][2]*100}%')
plt.show()