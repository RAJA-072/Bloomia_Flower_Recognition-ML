import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Directory path for the dataset
data_dir = r'C:\Users\raji\Downloads\Bloomia_Flower Recognition\archive (1)\flowers'

# Parameters
img_height, img_width = 150, 150  # Adjust based on your dataset
batch_size = 32

# Data Preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Split 20% for testing

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

test_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Define Models
def create_simple_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_data.num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_resnet50():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(train_data.num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_vgg16():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(train_data.num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Models to Test
models = {
    "Simple_CNN": create_simple_cnn,
    "ResNet50": create_resnet50,
    "VGG16": create_vgg16
}

# Directory to save models
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

# Training, Evaluation, and Saving Loop
results = {}
for model_name, model_func in models.items():
    model = model_func()
    print(f"\nTraining {model_name}...\n")
    model.fit(train_data, epochs=5, validation_data=test_data)  # Adjust epochs as needed

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_data)
    results[model_name] = test_accuracy
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")

    # Save the model
    model_path = os.path.join(save_dir, f"{model_name}.h5")
    model.save(model_path)
    print(f"{model_name} saved to {model_path}")

# Display results
print("\nModel Performance Comparison:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")

# Identify the best model
best_model = max(results, key=results.get)
print(f"\nBest model: {best_model} with accuracy {results[best_model]:.4f}")
