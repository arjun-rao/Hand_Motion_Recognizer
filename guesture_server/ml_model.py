import os.path
import numpy as np
from datetime import datetime
from collections import Counter
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Input
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model


def shuffle_train_data(x_train, y_train, num_features):
    shuffled_x = shuffle(*x_train, y_train)
    x_train = shuffled_x[:num_features]
    y_train = shuffled_x[num_features]
    return x_train, y_train

def get_train_data(events, feature_names, max_review_length):
    features = []
    y_train = []
    num_features = len(feature_names)
    for i in range(num_features):
        features.append([])
    for event in events.keys():
        for fname in range(events[event]):
            df = pd.read_csv(f'./RAW_Data/{event}_{fname}.csv')
            print(f'{event}_{fname}.csv')
            for idx, name in enumerate(feature_names):
                features[idx].append(np.array(df[name][:max_review_length]))
            y_train.append(event)
    return features, y_train, feature_names

def parse_train_data(features, num_samples, max_review_length):
    feature_list = [np.array(feature).reshape(num_samples, 1, max_review_length) for feature in features]
    data_train = np.array(feature_list)
    return data_train

def parse_test_data(fname, feature_names, max_review_length):
    df = pd.read_csv(fname)
    features = []
    num_features = len(feature_names)
    for i in range(num_features):
        features.append([])
    for idx, name in enumerate(feature_names):
        features[idx].append(np.array(df[name][:max_review_length]))
    data = [np.array(np.array(feature).reshape(1, 1, max_review_length)) for feature in features]
    return data

def encode_labels(y_train):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_train)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded, label_encoder

def invert_onehot(label, le):
    inverted = le.inverse_transform([np.argmax(label)])
    return inverted

def execute_layers(inputs, layers):
    """
    Computes each input to all given layers and returns their outputs in a list
    @param inputs : List of inputs to be passed through the given layer(s)
    @param layers : List of layers though which each input will be passed
    @return       : List of output generated from each input
    """
    outputs = []
    for _input in inputs:
        _output = _input
        for layer in layers:
            _output = layer(_output)
        outputs.append(_output)
    return outputs

def init_model(max_review_length, NUM_LABELS):
    input_x = Input(shape=(1, max_review_length), name="Acceleration_x")
    input_y = Input(shape=(1, max_review_length), name="Acceleration_y")
    input_z = Input(shape=(1, max_review_length), name="Acceleration_z")
    input_p = Input(shape=(1, max_review_length), name="pitch")
    input_r = Input(shape=(1, max_review_length), name="roll")
    input_frc = Input(shape=(1, max_review_length), name="force")
    input_flx = Input(shape=(1, max_review_length), name="flex")
    shared_layers = (LSTM(max_review_length, activation="tanh", name="Shared_LSTM", dropout=0.25),
                 Dense(NUM_LABELS*3*64,  activation="relu", name="Shared_Dense_1"),
                 Dense(NUM_LABELS*3*64,  activation="relu", name="Shared_Dense_2"),
                 Dense(NUM_LABELS*1*64,  activation="relu", name="Shared_Dense_3"))
    shared_output = execute_layers(
            inputs=(input_x, input_y, input_z, input_p, input_r, input_frc, input_flx), layers=shared_layers)
    concat      = keras.layers.concatenate(shared_output,name="Concatenate")
    dense_1     = Dense(39, activation="relu",    name="Dense_1")(concat)
    main_output = Dense(NUM_LABELS,   activation="softmax", name="Classification_Layer")(dense_1)
    model = Model(inputs=(input_x, input_y, input_z, input_p, input_r, input_frc, input_flx), outputs=main_output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    print("Model Summary", model.summary(), sep="\n")
    return model


def train_model(data_train, labels_train, max_review_length, num_labels):
    model = init_model(max_review_length, num_labels)
    history = model.fit(x=[*data_train], y=labels_train, epochs=10, batch_size=10)
    if not os.path.exists("Models"):
        os.makedirs("Models")
    file_path = os.path.join("Models", "latest.HDF5")
    model.save(file_path)
    return model, file_path

def load_trained_model(file_path):
    return load_model(file_path)

def predict_from_file(fname, model, feature_names, max_review_length, le):
    data = parse_test_data(fname, feature_names, max_review_length)
    return invert_onehot(model.predict(data)[0], le)[0]
