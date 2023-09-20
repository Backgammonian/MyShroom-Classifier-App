from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import tensorflow
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image
import os
import random
import string
import json
import skimage
import base64
from io import BytesIO

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_random_string(length = 15, chars = string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(length))

def load_model(path, loss, optimizer, metrics):
    model = keras.models.load_model(path, compile = False)
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    return model

def load_models_from_paths(paths):
    loss = 'sparse_categorical_crossentropy'
    optimizer = 'adam'
    metrics = ['accurary']
    models = []
    model_names = []
    for path in paths:
        _, extension = os.path.splitext(path)
        if (extension != '.keras'):
            continue
        models.append(load_model(path, loss, optimizer, metrics))
        model_names.append(os.path.splitext(os.path.basename(path))[0])
    return models, model_names

def get_image(image_path, image_width, image_height, num_channels):
    image = Image.open(image_path)
    image_resized = image.resize((image_width, image_height))
    image_reshaped = np.reshape(
        np.asarray(image_resized),
        (1, image_width, image_height, num_channels))
    return image_resized, image_reshaped
    
def test_model(model, label_encoder, test_image_reshaped):
    inference = model(test_image_reshaped, training = False).numpy()
    predicted_value = np.argmax(inference, axis = -1)
    predicted_label = label_encoder.inverse_transform(predicted_value)[0]
    probability = inference[0][predicted_value][0] * 100
    return predicted_value, predicted_label, probability

def encode(image):
    with BytesIO() as output_bytes:
        PIL_image = Image.fromarray(skimage.img_as_ubyte(image))
        PIL_image.save(output_bytes, 'JPEG')
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return base64_str

UPLOAD_FOLDER = './_uploads/'
MODELS_FOLDER = './_models/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
labels = [
    'Agaricus',
    'Amanita',
    'Boletus',
    'Cortinarius',
    'Entoloma',
    'Hygrocybe',
    'Lactarius',
    'Russula',
    'Suillus']
encoder = LabelEncoder()
encoder.fit(labels)
model_paths = os.listdir(MODELS_FOLDER)
for i in range (len(model_paths)):
    model_paths[i] = MODELS_FOLDER + model_paths[i]
models, model_names = load_models_from_paths(model_paths)
image_width = 128
image_height = 128
image_channels = 3
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 2**20 # 16 MB

@app.route('/')
def home():
    return render_template('index.html', results = {})

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No selected file!')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file!')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            original_file_name = secure_filename(file.filename)
            file_extension = os.path.splitext(original_file_name)[-1]
            file_name = get_random_string() + file_extension
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file.save(file_path)
            resized_image, reshaped_image = get_image(
                file_path,
                image_width,
                image_height,
                image_channels)
            predictions = []
            for i, model in enumerate(models):
                predicted_value, prediction_label, probability = test_model(
                    model, 
                    encoder,
                    reshaped_image)
                predictions.append((
                    prediction_label,
                    probability,
                    model_names[i]))
            return render_template(
                'index.html',
                 results = {'image': encode(resized_image), 'predictions': predictions})
    return render_template('index.html', results = {})

if __name__ == '__main__':
    # app.run(debug = True)
    app.run(debug = False, host = '0.0.0.0', port = '7000')