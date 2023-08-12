import matplotlib.pyplot as plt
import numpy as np

def show_images(x, y, title_str='Label'):
    for c in range(1, 10):
        plt.subplot(3, 3, c)
        i = np.random.randint(len(x)) # Generate random integer number
        im = x[i] # get i-th image
        plt.axis("off")
        label = y[i] # get i-th label
        plt.title("{} = {}".format(title_str, label))
        plt.imshow(im, cmap='Greys')