import io
import base64
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import subprocess
app = Flask(__name__)

# Load the pre-trained model
model = load_model('digit_recognition_model.keras')


def addBackground(image):
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        # Create a new image with a white background
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))

        # Paste the original image on top of the white background
        background.paste(image, (0, 0), image)

        # Convert the image to RGB mode to remove the alpha channel
        result = background.convert('RGB')

    else:
        # If the image doesn't have an alpha channel, just convert it to RGB
        result = image.convert('RGB')

    return result


def clearscreen():
    print("\n"*100)


def plot_base64_image(base64_data_url):
    # Split the data URL to get the base64 data
    header, base64_data = base64_data_url.split(',', 1)

    # Decode the base64 data
    image_data = base64.b64decode(base64_data)

    # Convert binary data to an image
    image = Image.open(io.BytesIO(image_data))

    # Convert image to NumPy array and plot
    image_array = np.array(image)
    plt.imshow(image_array, cmap='gray')
    plt.show()


def imagepart(image_data, view):
    img = Image.open(io.BytesIO(image_data))
    img = addBackground(img)
    img = ImageOps.grayscale(img)
    img = img.convert("L")
    img = img.resize((28, 28))
    img_array = np.asarray(img)
    img_array = img_array / 255.0
    img_array = 1.0 - img_array
    img_array = np.reshape(img_array, (28, 28, 1))  # Reshape to 3D array
    if (view):
        plt.imshow(img_array.squeeze(), cmap='gray')
        plt.title("Preprocessed Image")
        plt.show()
    return np.expand_dims(img_array, axis=0)  # Add batch dimension


def preprocess_image(img_data, view):
    # Split the data URL to get the base64 data
    header, base64_data = img_data.split(',', 1)

    # Decode the base64 data
    image_data = base64.b64decode(base64_data)

    image = imagepart(image_data, view)

    print("Image converted and preprocessed successfully.")
    return image  # Add batch dimension


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(f"data: {data}")
    img_data = data['image']

    img_array = preprocess_image(img_data, view=False)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return jsonify({'prediction': int(predicted_class)})


if __name__ == '__main__':
    clearscreen()
    app.run(debug=True)
