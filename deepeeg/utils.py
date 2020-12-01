import tensorflow as tf
import tensorflow.keras.backend as K


def hilbert(x):
    '''
    Implements the hilbert transform.
    Args:
        x: The input sequence. A tensor of shape (None, channel, sample, filter)
    Returns:
        xc: A complex sequence of the same shape.
    '''
    if len(x.shape) != 4:
        raise NotImplementedError
    if x.dtype != 'float32':
        x = tf.cast(x, dtype=tf.float32)

    if K.is_keras_tensor(x):
        if K.image_data_format() == 'channels_first':
            filter_num = K.int_shape(input_tensor)[1]
            channel = K.int_shape(input_tensor)[2]
            N = K.int_shape(input_tensor)[3]
        else:
            channel = K.int_shape(input_tensor)[1]
            N = K.int_shape(input_tensor)[2]
            filter_num = K.int_shape(input_tensor)[3]

    x = tf.transpose(x, [0, 1, 3, 2])
    x = tf.reshape(x, [-1, channel * filter_num, N])
    x = tf.complex(x, tf.zeros_like(x))
    Xf = tf.signal.fft(x)
    if N % 2 == 0:
        part0 = tf.ones(1)
        part1 = 2 * tf.ones(N // 2 - 1)
        part2 = tf.ones(1)
        part3 = tf.zeros(N // 2 - 1)
        h = tf.concat([part0, part1, part2, part3], axis=0)
    else:
        part0 = tf.ones(1)
        part1 = 2 * tf.ones((N + 1) // 2 - 1)
        part2 = tf.zeros((N + 1) // 2 - 1)
        h = tf.concat([part0, part1, part2], axis=0)

    hs = tf.expand_dims(h, 0)
    hs = tf.expand_dims(hs, 0)

    tf_hc = tf.complex(hs, tf.zeros_like(hs))
    xc = Xf * tf_hc
    out = tf.signal.ifft(xc)
    return tf.transpose(tf.reshape(out, [-1, channel, filter_num, N]),
                        [0, 1, 3, 2])
