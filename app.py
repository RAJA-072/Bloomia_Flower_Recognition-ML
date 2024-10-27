from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your trained model
model = load_model('saved_models/VGG16.h5')  # Replace with the path to your model

# Define a function for predicting the flower type
# Add this dictionary at the top of app.py
flower_classes = {
    4: "Tulip",
    3: "Sunflower",
    2: "Rose",
    1: "Dandelion",
    0: "Daisy"
}

# Update predict_flower function to return the flower name
def predict_flower(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return flower_classes.get(predicted_class, "Unknown")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Make prediction
    predicted_class = predict_flower(filepath)
    # Remove the uploaded image after prediction (optional)
    os.remove(filepath)

    # Return the result
    return jsonify({'prediction': str(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
