from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.regularizers import l1, l2

from ..layers import SincConv


def SEEGNet(nclass,
            channel_size,
            sample_size,
            sample_rate,
            kernel_size,
            F1,
            min_low_hz=1,
            min_band_hz=1,
            D=1,
            dropout_rate=0.2,
            l1_reg=0.001,
            l2_reg=0.1,
            *args,
            **kwargs):

    inputs = Input(shape=(channel_size, sample_size, 1))
    ##################################################################
    s_block = SincConv(filters=F1,
                       kernel_size=kernel_size,
                       sample_rate=sample_rate,
                       min_low_hz=min_low_hz,
                       min_band_hz=min_band_hz)(inputs)
    s_block = BatchNormalization(momentum=0.85)(s_block)
    s_block = DepthwiseConv2D((channel_size, 1),
                              use_bias=False,
                              depth_multiplier=D,
                              depthwise_regularizer=l2(l2_reg))(s_block)
    s_block = BatchNormalization(momentum=0.85)(s_block)

    c_block = Conv2D(int(F1 * D * 1.5),
                     1,
                     padding='same',
                     use_bias=False,
                     kernel_regularizer=l2(l2_reg))(s_block)
    c_block = BatchNormalization(momentum=0.85)(c_block)
    c_block = Activation('relu')(c_block)
    c_block = AveragePooling2D((1, 4))(c_block)
    c_block = Dropout(dropout_rate)(c_block)

    c_block = DepthwiseConv2D((1, 16), use_bias=False, padding='same')(c_block)
    c_block = BatchNormalization(momentum=0.85)(c_block)
    c_block = Activation('relu')(c_block)
    c_block = Dropout(dropout_rate)(c_block)
    c_block = Conv2D(F1 * D,
                     1,
                     padding='same',
                     use_bias=False,
                     kernel_regularizer=l2(l2_reg))(c_block)
    c_block = BatchNormalization(momentum=0.85)(c_block)
    c_block = Activation('relu')(c_block)
    c_block = AveragePooling2D((1, 8))(c_block)
    c_block = Dropout(dropout_rate)(c_block)

    x = Flatten(name='flatten')(c_block)
    x = Dense(nclass, name='dense')(x)
    softmax = Activation('softmax', name='softmax')(x)

    return Model(inputs=inputs, outputs=softmax)
