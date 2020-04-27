from flask import Flask, request, redirect, jsonify, render_template
import os
import json
import csv
from collections import defaultdict
import numpy as np
import traceback
from threading import Thread
import pickle
import tensorflow as tf
import keras

import ml_model

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

events = defaultdict(int)
preds = defaultdict(int)
model = None
firstLoad = False
mode_file_path = None
label_encoder = None
feature_names = ['x', 'y', 'z', 'p', 'r', 'f', 'flx']
max_review_length = 120
background_thread = None
session = tf.Session(graph=tf.Graph())

def try_loading_models():
    global model, label_encoder, mode_file_path
    mode_file_path = os.path.join("Models", "latest.HDF5")
    with session.graph.as_default():
        keras.backend.set_session(session)
        if (model is None or firstLoad==False) and os.path.isfile(mode_file_path):
            print('Loading saved model')
            model = ml_model.load_trained_model(mode_file_path)
            label_encoder = pickle.load(open('./Models/latest_le.pkl', 'rb'))
            return model, label_encoder
    return model, label_encoder

def train_model():
    global model, mode_file_path, label_encoder, background_thread
    with session.graph.as_default():
        keras.backend.set_session(session)
        try:
            x_train, y_train, f = ml_model.get_train_data(events, feature_names, max_review_length)
            x_train, y_train = ml_model.shuffle_train_data(x_train, y_train, len(feature_names))
            labels_train, le = ml_model.encode_labels(y_train)
            NUM_SAMPLES = len(labels_train)
            NUM_LABELS = len(np.unique(y_train))
            data_train = ml_model.parse_train_data(x_train, NUM_SAMPLES, max_review_length)
            print(f'Training model for dataset: {data_train.shape}')
            print(f'Labels: {np.unique(y_train)}')
            model_binary, model_path = ml_model.train_model(data_train, labels_train, max_review_length, NUM_LABELS)
            model = model_binary
            mode_file_path = model_path
            label_encoder = le
            pickle.dump(label_encoder, open('./Models/latest_le.pkl', 'wb'))
            json.dump(events, open('events.json', 'w'))
            print("Training Complete")
            return True
        except:
            traceback.print_exc()
            return False


def predict(fname):
    global model, mode_file_path, label_encoder, firstLoad
    with session.graph.as_default():
        keras.backend.set_session(session)
        if not firstLoad:
            model, label_encoder = try_loading_models()
            firstLoad = True
        if not os.path.isfile(mode_file_path):
            if background_thread is not None:
                background_thread.join()
        if model is not None:
            if label_encoder is None:
                label_encoder = pickle.load(open('./Models/latest_le.pkl', 'rb'))
            try:
                result = ml_model.predict_from_file(fname, model, feature_names, max_review_length, label_encoder)
            except:
                try_loading_models()
                result = ml_model.predict_from_file(fname, model, feature_names, max_review_length, label_encoder)
            print(result)
            return result
        print('Model Not Trained Yet')
    return ''

@app.route('/api/upload_train', methods = ['POST'])
def upload_train():
    body = request.form.to_dict()
    g_id = body['g_id']
    data = json.loads(body['data'])
    print(f'Received data for training: {g_id}')
    with open(f'./RAW_data/{g_id}_{events[g_id]}.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(feature_names)
        writer.writerows(data)
    events[g_id] += 1
    return jsonify({'status' : f'OK:{events[g_id]}'}), 200


@app.route('/api/start_train', methods = ['POST'])
def start_train():
    global background_thread
    print('Received train request')

    if len(events) != 0:
        if background_thread is not None and background_thread.isAlive():
            return jsonify({'status' : f'Training request in progress...'}), 200
        background_thread = Thread(target=train_model)
        background_thread.start()
        # background_thread.join()
        # train_model()
        return jsonify({'status' : f'New Training request received'}), 200
    else:
        return jsonify({'status' : f'Not enough data to train'}), 200

@app.route('/api/get_prediction', methods = ['POST'])
def get_prediction():
    body = request.form.to_dict()
    g_id = body['g_id']
    data = json.loads(body['data'])
    print(f'Received data for prediction: {g_id}')
    file_name = f'./RAW_data/predict/{g_id}_{preds[g_id]}.csv'
    with open(file_name, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(feature_names)
        writer.writerows(data)
    preds[g_id] += 1
    if background_thread is not None and background_thread.is_alive():
        background_thread.join()
    result = predict(file_name)
    if result != '':
        return jsonify({'status' : f'{result}'}), 200
    else:
        return jsonify({'status' : f'No Model Trained For Prediction'}), 200


@app.route('/')
def index():
    user_agent = request.headers.get('User-Agent')
    return '<p>Your browser is %s</p>' % user_agent

if __name__ == '__main__':
    if os.path.isfile('events.json'):
        event_list = json.load(open('events.json', 'r'))
        for key, value in event_list.items():
            events[key] = value
        print(f'Data: {events}')
        try_loading_models()
    app.run(host = '0.0.0.0', port = 5001, threaded=False)