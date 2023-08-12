import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Testing a real-world image
image_path = "../images/Burmese_number_5.jpg"
img_input = tf.io.read_file(image_path)
img_input = tf.image.decode_image(img_input)
img_input = tf.image.rgb_to_grayscale(img_input)
img_input = 255 - img_input
img_input = tf.image.resize(img_input, size=(28, 28), method='bilinear')

plt.imshow(img_input, cmap='Greys')

img_input = tf.expand_dims(img_input, axis=0)

# Load model
loaded_model = tf.keras.models.load_model("../model")

y_input_pred = loaded_model.predict(img_input)
print(np.argmax(y_input_pred))