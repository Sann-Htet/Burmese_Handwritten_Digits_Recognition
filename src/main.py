import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import create_deeper_model
from evaluation import evaluate_model

local_filename = '../dataset/data.pkl'
dataset = []

with open(local_filename, "rb") as file:
    # Deserializing which is a transforming serialized data back into its original format
    dataset = pickle.load(file)
            
trainDataset = dataset["trainDataset"]
testDataset = dataset["testDataset"]

X_train = np.array([td["image"] for td in trainDataset])
y_train = np.array([td["label"] for td in trainDataset]) 
X_test = np.array([td["image"] for td in testDataset])
y_test = np.array([td["label"] for td in testDataset])

# Reshape images
X_train = X_train.astype(np.float32).reshape(len(X_train), 28, 28, 1)
y_train = y_train.reshape(len(y_train), 1)
X_test = X_test.astype(np.float32).reshape(len(X_test), 28, 28, 1)
y_test = y_test.reshape(len(y_test), 1)

# Split into training, validation, and test sets

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = create_deeper_model()

# Configures the model for training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
epoch = 10
history = model.fit(X_train, y_train,
                    epochs=epoch,
                    batch_size=128,
                    validation_data=(X_val, y_val))

# Evaluate the model
accuracy = evaluate_model(model, X_test, y_test)
print("Test accuracy:", accuracy)

# Save the model

# save the model using "SavedModel" format
#model.save('../model')

# save the model using "HDF5" format
#model.save('../model.h5')