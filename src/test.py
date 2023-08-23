import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load model via "saved_model" format
model = tf.keras.models.load_model('../model')

# Testing a real-world image
image_path = "../images/NRC_digits/five.png"
image = tf.keras.preprocessing.image.load_img(image_path) # load image
image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY) # convert to Grayscale
image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR) # rescale the image
# Apply thresholding to remove noise and enhance contrast
threshold_value = 107  # Adjust this threshold as needed
_, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

image = tf.expand_dims(thresholded_image, axis=-1)
image = tf.cast(image, tf.float32)
image = 255 - np.array(image)

plt.imshow(image, cmap="Greys")

img_input = tf.expand_dims(image, axis=0)

y_input_pred = model.predict(img_input)
print(np.argmax(y_input_pred))
print("Achieved a {:.2%} accuracy rate.".format(np.max(y_input_pred)))