import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.python.keras.regularizers import l2


class SincConv(Layer):
    """Sinc-based convolution

    Input shape
    - 4D tensor with shape:``(batch_size,channel_size,sample_size,1)``.

    Output shape
    - 4D tensor with shape:``(batch_size,channel_size,sample_size,filters)``.
    
    Parameters
    ----------
    filters : `int`, Number of filters.
    kernel_size : `int`, Filter length.
    sample_rate : `int`, Sample rate.
    min_low_hz : 'int', Min low pass filter.
    min_band_hz : 'int', Min band pass length.
    seed : 'int', Seed for init weight.
    l2_reg : 'float' between 0 and 1, L2 regularizer strength.

    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """
    @staticmethod
    def to_mel(hz):
        return np.log10(1 + hz / 3)

    @staticmethod
    def to_hz(mel):
        return 3 * (10**mel - 1)

    def __init__(self,
                 filters,
                 kernel_size,
                 sample_rate,
                 min_low_hz=1,
                 min_band_hz=4,
                 **kwargs):

        super(SincConv, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

    def build(self, input_shape):

        if len(input_shape) < 4 or input_shape[-1] != 1:
            raise ValueError(
                'A `SincConv` layer should be called on a 4D tensor '
                'with shape:``(batch_size,1,channel_size,sample_size)')

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = self.min_low_hz
        # high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        high_hz = 45 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz),
                          self.filters + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (filters, 1)
        self.low_hz_ = self.add_weight(
            name='low_hz',
            shape=(1, self.filters),
            trainable=True)
        self.band_hz_ = self.add_weight(
            name='band_hz',
            shape=(1, self.filters),
            trainable=True)

        # filter lower frequency (1, filters)
        self.low_hz_.assign(
            tf.cast(tf.reshape(hz[:-1], [1, -1]), dtype=tf.float32))

        # filter frequency band (1, filters)
        self.band_hz_.assign(
            tf.cast(tf.reshape(np.diff(hz), [1, -1]), dtype=tf.float32))

        # Hamming window
        # computing only half of the window
        n_lin = tf.linspace(0., (self.kernel_size / 2) - 1,
                            num=int(self.kernel_size / 2))
        self.window_ = tf.reshape(
            0.54 - 0.46 * tf.math.cos(2 * math.pi * n_lin / self.kernel_size),
            [-1, 1])

        # (kernel_size/2, 1)
        n = (self.kernel_size - 1) / 2.0
        # Due to symmetry, I only need half of the time axes
        self.n_ = 2 * math.pi * tf.reshape(tf.range(-n, 0),
                                           [-1, 1]) / self.sample_rate

        super(SincConv,
              self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 4:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 4 dimensions" %
                (K.ndim(inputs)))

        # (1, filters)
        low = self.min_low_hz + tf.math.abs(self.low_hz_)
        high = tf.clip_by_value(
            low + self.min_band_hz + tf.math.abs(self.band_hz_),
            self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[0]

        # (kernel_size/2, filters)
        f_times_t_low = tf.matmul(self.n_, low)
        f_times_t_high = tf.matmul(self.n_, high)

        # # Equivalent of Eq.4 of the reference paper
        # # (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET).
        # # I just have expanded the sinc and simplified the terms.
        # # This way I avoid several useless computations.
        band_pass_left = self.window_ * (
            (tf.math.sin(f_times_t_high) - tf.math.sin(f_times_t_low)) /
            (self.n_ / 2))
        band_pass_center = 2 * tf.reshape(band, [1, -1])
        band_pass_right = tf.reverse(band_pass_left, axis=[0])

        band_pass = tf.concat(
            [band_pass_left, band_pass_center, band_pass_right], axis=0)
        band_pass = band_pass / (2 * band[None, :])

        filters = tf.reshape(band_pass, [1, self.kernel_size, 1, self.filters])

        return tf.nn.conv2d(inputs, filters, strides=1, padding='SAME')

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0], input_shape[1], self.filters)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'sample_rate': self.sample_rate,
            'min_low_hz': self.min_low_hz,
            'min_band_hz': self.min_band_hz
        }
        base_config = super(SincConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

