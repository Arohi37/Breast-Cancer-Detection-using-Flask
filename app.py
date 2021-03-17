from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from PIL import Image
from io import BytesIO
import numpy as np

app = Flask(__name__)
model = None
graph = tf.compat.v1.get_default_graph()

def load_request_image(image):
    image = Image.open(BytesIO(image))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((50, 50))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    return image

def load_model():
    with graph.as_default():
        json_file = open('./model/model.json', 'r')
        model_json = json_file.read()
        json_file.close()
        global model
        model = model_from_json(model_json)
        model.load_weights("./model/weights.h5")

def predict_class(image_array):
    classes = ["Benign", "Malignant"]
    with graph.as_default():
        y_pred = model.predict(image_array, batch_size=None, verbose=0, steps=None)[0]
        probab = np.argmax(y_pred, axis=0)
        probab = y_pred[probab]
        
        if probab<=0.5:
            class_predicted, confidence = classes[0],  100-probab*100
        else:
             class_predicted, confidence = classes[1],probab*100
        return class_predicted, confidence




@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"].read()
    image = load_request_image(image)
    class_predicted,confidence = predict_class(image)  
    image_class = { "class": class_predicted, "confidence": str(confidence) } 
    return jsonify(image_class)

if __name__ == "__main__":
    load_model()
    app.run(debug = False, threaded = False)

if __name__ == "app":
    load_model()
