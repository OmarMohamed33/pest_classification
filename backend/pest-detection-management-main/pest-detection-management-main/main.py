# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
     #Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
class_labels = [
    'ants',
    'bees',
    'beetle',
    'catterpillar',
    'earthworms',
    'earwig',
    'grasshopper',
    'moth',
    'slug',
    'snail',
    'wasp',
    'weevil'
]

insects_and_invertebrates = {
    'ants': 'Small social insects known for their organized colonies and ability to carry objects many times their own weight.',
    'bees': 'Flying insects known for their role in pollination and for producing honey and beeswax.',
    'beetle': 'A group of insects with hard, shell-like wings and varied shapes and sizes.',
    'catterpillar': 'The larval stage of butterflies and moths, known for their segmented bodies and voracious appetite for leaves.',
    'earthworms': 'Soil-dwelling annelids that play a key role in aerating the soil and breaking down organic matter.',
    'earwig': 'Insects with characteristic pincers on their abdomen and nocturnal habits.',
    'grasshopper': 'Leaping insects known for their powerful hind legs and ability to produce sound by rubbing their wings or legs together.',
    'moth': 'Nocturnal flying insects closely related to butterflies, often attracted to light.',
    'slug': 'Soft-bodied, shell-less mollusks that thrive in moist environments and feed on plants.',
    'snail': 'Mollusks with a coiled shell, known for their slow movement and mucus trails.',
    'wasp': 'Insects related to bees and ants, often with a sting and sometimes forming colonies.',
    'weevil': 'Beetles characterized by their elongated snouts, known for infesting stored grains.'
}


UPLOAD_FOLDER = 'img'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model_path = 'saved_model/trained_model(1).h5'
if os.path.exists(model_path):
    try:
        # Load the model
        MODEL = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"No directory found at {model_path}")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload/img', methods=['POST'])

def upload_file():
    # Check if the post request has the file part
    data = {}
    data['error'] = 0
    print("\n")
    data['message'] = ""
    print("\n")
    data['name'] = ""
    print("\n")
    if 'file' not in request.files:
        data['message'] = 'No file part'
        return jsonify( data)

    file = request.files['file']
    # If the user does not select a file, the browser submits an empty
    # part without filename
    if file.filename == '':
        data['message'] = 'No selected file'
        data['error'] = 1
        return jsonify( data)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        imgPath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save( imgPath)

        img_array = preprocess_image(imgPath,(224, 224))
        # Classify the image
        predicted_class_label, predicted_class_probability = classify_image(MODEL, img_array, class_labels)


        data['message'] = 'File successfully uploaded'
        print("\n")
        data['name'] = predicted_class_label
        # data['percentage'] = str(int(predicted_class_probability*100))+"%"
        print("\n")
        data['description'] = insects_and_invertebrates[predicted_class_label]
        print('\n')
        print(data)
        return jsonify( data)
    else:
        data['message'] = 'File type not allowed'
        data['error'] = 2
        return jsonify( data)


def preprocess_image(img_path, target_size):
    """
    Preprocess the input image to match the format expected by the model.

    Parameters:
    - img_path: str, path to the input image
    - target_size: tuple, target size to resize the image (height, width)

    Returns:
    - img_array: numpy array, preprocessed image ready for model prediction
    """
    # Load the image from the file path
    img = image.load_img(img_path, target_size=target_size)

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Expand the dimensions of the array to match the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image (e.g., normalization)
    img_array = preprocess_input(img_array)

    return img_array


def classify_image(model, img_array, class_labels):
    """
    Classify the preprocessed image using the loaded model.

    Parameters:
    - model: the loaded Keras model
    - img_array: numpy array, preprocessed image
    - class_labels: list of str, class labels in the order corresponding to the model's output

    Returns:
    - predicted_class_label: str, the predicted class label
    - predicted_class_probability: float, the probability of the predicted class
    """
    predictions = model.predict(img_array)
    print(f'Predictions: {predictions}')
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    print(f'Predicted class index: {predicted_class_index}')

    # Ensure the predicted class index is within the range of class_labels
    if predicted_class_index < len(class_labels):
        predicted_class_label = class_labels[predicted_class_index]
        predicted_class_probability = predictions[0][predicted_class_index]
    else:
        raise ValueError("The predicted class index is out of range of the class labels.")

    return predicted_class_label, predicted_class_probability

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if __name__ == '__main__':
    app.run(debug=True)