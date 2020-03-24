import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from deepeeg.models import SincEEGNet, EEGNet, EEGSENet


def preprocess(data, label):

    data = tf.cast(data, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.int32)

    std = tf.math.reduce_std(data, axis=0)
    mean = tf.math.reduce_mean(data, axis=0)
    data = (data - mean) / std

    return data, label


def test_mi():
    dict_name = 'data\\mi\\'
    data_name = list(filter(lambda x: x.startswith('S'),
                            os.listdir(dict_name)))

    data = None
    label = None

    for sub in data_name:
        with np.load(dict_name + sub) as f:
            if data is None:
                data = f['data']
                label = f['label']
            else:
                data = np.concatenate((data, f['data']))
                label = np.concatenate((label, f['label']))

    channel_size = data.shape[1]
    sample_size = data.shape[2]
    sample_rage = 160

    data = np.reshape(data, [-1, channel_size, sample_size, 1])
    oe = OneHotEncoder()
    label = oe.fit_transform(label.reshape(-1, 1)).toarray()

    kf = KFold(n_splits=5)

    for train_index, test_index in kf.split(data):
        train_X, train_y = data[train_index], label[train_index]
        test_X, test_y = data[test_index], label[test_index]

        db_train = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        db_train = db_train.shuffle(1000).map(preprocess).batch(8)
        db_test = tf.data.Dataset.from_tensor_slices((test_X, test_y))
        db_test = db_test.shuffle(1000).map(preprocess).batch(8)

        model = EEGNet(nclass=2,
                       channel_size=channel_size,
                       sample_size=sample_size,
                       kernel_size=128,
                       F1=96,
                       F2=96)

        model.compile(
            "adam",
            "categorical_crossentropy",
            metrics=['AUC', 'acc'],
        )

        model.fit(
            db_train,
            validation_data=db_test,
            epochs=10,
            verbose=2,
        )


def test_delay_eegnet(file_name):
    with np.load(file_name) as f:
        data = f['data']
        label = f['label']

    model_name = "eegnet-{}".format(datetime.now().strftime('%b-%d-%H-%M'))
    log_dir = os.path.join(os.getcwd(), 'logs', model_name)
    model_path = os.path.join(
        os.getcwd(), 'model_save',
        'eegnet-{F1}-{F2}-{kernel}.h5'.format(F1=96, F2=96, kernel=63))

    channel_size = data.shape[1]
    sample_size = data.shape[2]
    sample_rage = 128

    train_X, test_X, train_y, test_y = train_test_split(data,
                                                        label,
                                                        test_size=0.1,
                                                        random_state=42)

    db_train = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    db_train = db_train.shuffle(1000).map(preprocess).batch(8)
    db_test = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    db_test = db_test.shuffle(1000).map(preprocess).batch(8)

    # model = SincEEGNet(nclass=3,
    #                    channel_size=channel_size,
    #                    sample_size=sample_size,
    #                    sample_rate=sample_rage,
    #                    kernel_size=129,
    #                    F1=20,
    #                    F2=96)

    model = EEGNet(nclass=3,
                   channel_size=channel_size,
                   sample_size=sample_size,
                   kernel_size=63,
                   F1=96,
                   F2=96)
    model.compile(
        "adam",
        "categorical_crossentropy",
        metrics=['AUC', 'acc'],
    )
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_callback = ModelCheckpoint(model_path)
    reduce_callback = ReduceLROnPlateau(monitor='val_loss', patience=5)
    model.fit(
        db_train,
        validation_data=db_test,
        epochs=20,
        verbose=1,
        callbacks=[tensorboard_callback, model_callback, reduce_callback])


def save_data():
    sub_name = os.listdir('data\\delay')

    data = []
    label = []

    for sub in sub_name:
        with np.load('data\\delay\\' + sub + '\\pre.npz') as f:
            data.append(f['data'])
            label.append(f['label'])

    data = np.concatenate(data)
    label = np.concatenate(label)

    channel_size = data.shape[1]
    sample_size = data.shape[2]

    data = np.reshape(data, [-1, channel_size, sample_size, 1])
    oe = OneHotEncoder()
    label = oe.fit_transform(label.reshape(-1, 1)).toarray()

    np.savez(r'G:\eeg\data\delay\all_3class_', data=data, label=label)


def test_eegsenet():
    dict_name = 'data\\mi\\'
    data_name = list(filter(lambda x: x.startswith('S'),
                            os.listdir(dict_name)))

    data = None
    label = None

    for sub in data_name:
        with np.load(dict_name + sub) as f:
            # if data is None:
                # data = f['data']
                # label = f['label']
            # else:
            #     data = np.concatenate((data, f['data']))
            #     label = np.concatenate((label, f['label']))

            data = f['data']
            label = f['label']
            if data.shape[0] < 8: continue

        channel_size = data.shape[1]
        sample_size = data.shape[2]
        sample_rage = 160

        data = np.reshape(data, [-1, channel_size, sample_size, 1])
        oe = OneHotEncoder()
        label = oe.fit_transform(label.reshape(-1, 1)).toarray()

        train_X, test_X, train_y, test_y = train_test_split(data,
                                                            label,
                                                            test_size=0.2,
                                                            random_state=42)

        db_train = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        db_train = db_train.shuffle(1000).map(preprocess).batch(4)
        db_test = tf.data.Dataset.from_tensor_slices((test_X, test_y))
        db_test = db_test.shuffle(1000).map(preprocess).batch(4)

        model_name = "eegnet-{}".format(sub.split('.')[0])
        log_dir = os.path.join(os.getcwd(), 'logs', model_name)
        model_path = os.path.join(os.getcwd(), 'model_save',
                                  model_name + '.h5')

        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        model_callback = ModelCheckpoint(model_path)
        reduce_callback = ReduceLROnPlateau(monitor='val_loss', patience=5)

        model = EEGSENet(nclass=2,
                         channel_size=channel_size,
                         sample_size=sample_size,
                         kernel_size=63,
                         reduction_ratio=16,
                         F1=32,
                         F2=32)
        # model.summary()
        model.compile(
            "adam",
            "categorical_crossentropy",
            metrics=['AUC', 'acc'],
        )

        model.fit(
            db_train,
            validation_data=db_test,
            epochs=10,
            verbose=1,
            callbacks=[tensorboard_callback, model_callback, reduce_callback])


if __name__ == "__main__":
    # test_delay_eegnet(r'G:\eeg\data\delay\all_3class.npz')
    # save_data()
    test_eegsenet()