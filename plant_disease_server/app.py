# coding=utf-8

from __future__ import division, print_function

import os
import cv2
import json
import logging
import numpy as np
from keras.models import load_model
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
from flask import Flask, redirect, url_for, request, render_template

logging.basicConfig(format='%(asctime)s %(levelname)s [%(module)s] %(message)s', level=logging.INFO)
log = logging.getLogger()

app = Flask(__name__)

with open("classes.json", "r") as fr:
    CLASS_INDEX = json.load(fr, encoding="utf-8")


def set_gpu(gpu_memory_frac=0.3):
    import tensorflow as tf
    import keras.backend as K
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_frac
    sess = tf.Session(config=config)
    K.set_session(sess)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_gpu(gpu_memory_frac=0.2)

MODEL_PATH = 'models/m7032_dn169v6_04_l.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
log.info('Model loaded. Start serving...\n Check http://127.0.0.1:5000/')

# You can also use pretrained model from Keras
# from keras.applications.resnet50 import ResNet50

# model = ResNet50(weights='imagenet')
# print('Model loaded. Check http://127.0.0.1:5000/')


def resize(image, test=False):
    """对所有图像做黑边填充"""
    w, h = image.shape[:2]
    if w == h:
        return image
    else:
        if w > h:
            w1 = np.random.randint(h + 1, w + 1)
            pad = np.random.randint(0, w1 - h)
            if test:
                w1 = w
                pad = int(0.5 * (w - h))
            new_image = np.zeros([w, w1, 3], dtype=np.uint8)
            new_image[0:w, pad:pad + h] = image
        else:
            h1 = np.random.randint(w + 1, h + 1)
            pad = np.random.randint(0, h1 - w)
            if test:
                h1 = h
                pad = int(0.5 * (h - w))
            new_image = np.zeros([h1, h, 3], dtype=np.uint8)
            new_image[pad:pad + w, 0:h] = image
    return new_image


def model_predict(img_path, model):
    img = cv2.imread(img_path)[:, :, ::-1]
    img = resize(img, test=True)
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(np.array(img, np.float32), axis=0) / 255.
    preds = model.predict(x)
    return preds


def post_process(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 61:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 18)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        with open("classes.json", "r") as fr:
            CLASS_INDEX = json.load(fr, encoding="utf-8")

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    log.info("results:\n {}".format(results))
    return results


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    data = {"success" : False}
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=5)   # ImageNet Decode
        pred_class = post_process(preds)
        # result = pred_class[0][0][1].encode("utf-8")               # Convert to string
        result = (pred_class[0][0][0] + "\t" + pred_class[0][0][1]).encode("utf-8")               # Convert to string
        if pred_class[0][0][2] < 0.5:
            result = "不认识"
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()


