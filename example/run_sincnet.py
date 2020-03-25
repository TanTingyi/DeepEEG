import sys
sys.path.append(r'G:\eeg\DeepEEG')

import os
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from deepeeg.models import SincEEGNet, EEGNet, EEGSENet

train_pramters = {
    'batch_size': 4,
    'epochs': 10,
    'model_name': "mymodel-{}".format(datetime.now().strftime("%m%d-%H%M")),
    'sample_rate': 160
}


def preprocess(data, label):

    data = tf.cast(data, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.int32)

    std = tf.math.reduce_std(data, axis=0)
    mean = tf.math.reduce_mean(data, axis=0)
    data = (data - mean) / std

    return data, label


def load_data(path_list):
    data = None
    label = None
    for path in path_list:
        with np.load(path) as f:
            if data is None:
                data = f['data']
                label = f['label']
            else:
                data = np.concatenate((data, f['data']))
                label = np.concatenate((label, f['label']))
    return data, label


def save_data(path_list):

    data = []
    label = []

    for path in path_list:
        with np.load(path) as f:
            data.append(f['data'])
            label.append(f['label'])

    data = np.concatenate(data)
    label = np.concatenate(label)

    channel_size = data.shape[1]
    sample_size = data.shape[2]

    data = np.reshape(data, [-1, channel_size, sample_size, 1])
    oe = OneHotEncoder()
    label = oe.fit_transform(label.reshape(-1, 1)).toarray()

    np.savez(r'G:\eeg\data\mi\all', data=data, label=label)


def test_eegsenet(path_list):
    data, label = load_data(path_list)
    channel_size = data.shape[1]
    sample_size = data.shape[2]
    data = np.reshape(data, [-1, channel_size, sample_size, 1])
    oe = OneHotEncoder()
    label = oe.fit_transform(label.reshape(-1, 1)).toarray()
    train_X, test_X, train_y, test_y = train_test_split(data,
                                                        label,
                                                        test_size=0.1,
                                                        random_state=42)

    db_train = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    db_train = db_train.shuffle(1000).map(preprocess).batch(
        train_pramters['batch_size'])
    db_test = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    db_test = db_test.shuffle(1000).map(preprocess).batch(
        train_pramters['batch_size'])
    #####################################################################
    log_dir = os.path.join(os.getcwd(), 'logs', train_pramters['model_name'])
    if os.path.exists(log_dir): os.rmdir(log_dir)
    model_path = os.path.join(os.getcwd(), 'model_save',
                              train_pramters['model_name'] + '.h5')

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_callback = ModelCheckpoint(model_path,
                                     monitor='val_acc',
                                     save_best_only=True)
    reduce_callback = ReduceLROnPlateau(monitor='val_loss', patience=3)

    model = EEGSENet(nclass=2,
                     channel_size=channel_size,
                     sample_size=sample_size,
                     kernel_size=128,
                     reduction_ratio=16,
                     F1=96,
                     F2=96)

    json_string = model.to_json()  
    with open(
            os.path.join(os.getcwd(), 'model_save',
                         train_pramters['model_name'] + '.json'),
                         'w') as f:
        json.dump(json_string, f)

    # model.summary()
    model.compile(
        "adam",
        "categorical_crossentropy",
        metrics=['AUC', 'acc'],
    )

    model.fit(
        db_train,
        validation_data=db_test,
        epochs=train_pramters['epochs'],
        verbose=1,
        callbacks=[tensorboard_callback, model_callback, reduce_callback])


if __name__ == "__main__":
    dict_name = r'G:\eeg\data\mi'
    data_name = list(filter(lambda x: x.startswith('S'),
                            os.listdir(dict_name)))
    # dict_name = r'G:\eeg\data\dilay'
    # data_name = ['all_3class.npz']
    
    path_list = [os.path.join(dict_name, name) for name in data_name]

    # test_delay_eegnet(r'G:\eeg\data\delay\all_3class.npz')
    # save_data(path_list)

    test_eegsenet(path_list)