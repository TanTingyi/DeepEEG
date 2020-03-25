from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten, Add
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2
import tensorflow as tf


def SqueezeExcitation(input, reduction_ratio):
    'Input must be channel last'
    num_filters = input.shape[-1]
    pool = GlobalAveragePooling2D()(input)
    squeeze = Dense(num_filters // reduction_ratio, activation='relu')(pool)
    excitation = Dense(num_filters, activation='sigmoid')(squeeze)
    scale = input * tf.reshape(excitation, [-1, 1, 1, num_filters])
    return scale


def residual_block(input):
    'Input must be channel last'
    input_channels = input.shape[-1]
    residual = BatchNormalization()(input)
    residual = Activation('elu')(residual)
    residual = SeparableConv2D(input_channels, (1, 12), padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
    residual = BatchNormalization()(residual)
    residual = Activation('elu')(residual)
    residual = SeparableConv2D(input_channels, (1, 12), padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
    residual = Add()([input, residual])
    return residual

def EEGSENet(nclass,
             channel_size,
             sample_size,
             kernel_size,
             reduction_ratio,
             dropoutRate=0.5,
             F1=96,
             D=1,
             F2=96):

    input1 = Input(shape=(channel_size, sample_size, 1))
    ##################################################################
    block1 = Conv2D(F1, (1, kernel_size), padding='same',
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((channel_size, 1),
                             use_bias=False,
                             depth_multiplier=1,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = Dropout(0.2)(block1)

    for _ in range(1):
        block2 = residual_block(block1)

    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(0.2)(block2)

    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nclass, name='dense')(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)
