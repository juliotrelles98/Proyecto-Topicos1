import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'Proyecto Topicos', 'modelo', 'modelo.h5')
model = None
labels = ['clase_0', 'clase_1', 'clase_2', 'clase_3', 'clase_4']

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = image.load_img(file, target_size=(150, 150))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    if model:
        preds = model.predict(x)
        idx = int(np.argmax(preds[0]))
        label = labels[idx] if idx < len(labels) else str(idx)
        confidence = float(preds[0][idx])
    else:
        label = 'modelo_no_disponible'
        confidence = 0.0

    return jsonify({'label': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
