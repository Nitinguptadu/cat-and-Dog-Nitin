import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os

app = Flask(__name__ ,template_folder='static')
port = int(os.environ.get("PORT", 5000))

@app.route('/')
def index():
    return render_template('predict.html')
def get_model():
    global model
    model = load_model('model/model.h5')
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print(" * Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(160, 160))
    
    prediction = model.predict(processed_image).tolist()
    print(prediction)
    response = {
        'prediction': {
            'dog': prediction[0][0],
            'cat': prediction[0][1]
        }
    }
    response = jsonify(response)
    response.headers.add('Access-Control-Allow-Origin', 'https://nitin0621.herokuapp.com/')
    return response


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0',port=port)
