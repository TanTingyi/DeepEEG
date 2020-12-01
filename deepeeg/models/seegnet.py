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
    sinc = SincConv(filters=F1,
                    kernel_size=kernel_size,
                    sample_rate=sample_rate,
                    min_low_hz=min_low_hz,
                    min_band_hz=min_band_hz)(inputs)
    sinc = LayerNormalization(beta_regularizer=l1(l1_reg))(sinc)
    block1 = DepthwiseConv2D((channel_size, 1),
                             use_bias=False,
                             depth_multiplier=D,
                             depthwise_regularizer=l2(l2_reg))(sinc)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropout_rate)(block1)

    block1 = Conv2D(F1 * D, 1, use_bias=False)(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)
    block1 = Dropout(dropout_rate)(block1)

    block2 = SeparableConv2D(F1 * D, (1, 16), use_bias=False,
                             padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropout_rate)(block2)

    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nclass, name='dense')(flatten)
    sigmoid = Activation('softmax', name='softmax')(dense)

    return Model(inputs=inputs, outputs=sigmoid)
