from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
app.config["UPLOAD_FOLDER1"] = "./static"
CORS(app)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    img = request.files['file']
    img.save(os.path.join('./static/' + img.filename))
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image = Image.open(os.path.join("./static/" + img.filename))
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.uint8)
    image = np.expand_dims(image, axis=0)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)
    print(output)
    return jsonify({'class': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
