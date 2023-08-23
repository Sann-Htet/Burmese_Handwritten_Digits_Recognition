from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from data_preprocessing import CustomPreprocessNormalizationLayer

def create_model():
    # Create a Sequential model
    model = Sequential()

    # Add the CustomPreprocessNormalizationLayer as the first layer
    model.add(CustomPreprocessNormalizationLayer())

    ### Add Convolutional and MaxPooling layers
    
    # CONV => RELU => MAX-POOLING
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    
    # CONV => RELU => MAX-POOLING => CONV
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Flatten the output for Dense layers
    model.add(Flatten())

    # Add Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer with a dropout rate of 0.5

    model.add(Dense(10, activation='softmax'))
    
    return model

def create_deeper_model():
    model = Sequential()

    # Add the CustomPreprocessNormalizationLayer as the first layer
    model.add(CustomPreprocessNormalizationLayer())

    # Convolutional layer block 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    # Convolutional layer block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    # Convolutional layer block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    # Flatten the output for Dense layers
    model.add(Flatten())

    # Dense layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(10, activation='softmax'))

    return model