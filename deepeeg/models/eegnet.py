from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm


def EEGNet(nclass,
           channel_size,
           sample_size,
           kernel_size,
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
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(0.2)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False,
                             padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(0.2)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nclass, name='dense')(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)
