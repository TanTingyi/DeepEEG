import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Input, Add
from tensorflow.keras.layers import Activation, Lambda, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import AveragePooling2D, AveragePooling1D
from tensorflow.keras.layers import SeparableConv2D, Concatenate
from tensorflow.keras.layers import DepthwiseConv2D, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow import keras

from ..layers import SincConv


class DilatedBlock(Model):
    """
    Creates a single causal dilated convolution layer

                |-> [gate]   -|        |-> 1x1 conv -> skip output
                |             |-> (*) -|
         input -|-> [filter] -|        |-> 1x1 conv -|
                |                                    |-> (+) -> dense output
                |------------------------------------|
    """
    def __init__(self, dilation, output_width, residual_channels,
                 dilation_channels, skip_channels, use_biases, regularizer,
                 last_layer, **kwargs):
        super(DilatedBlock, self).__init__(**kwargs)
        self.output_width = output_width
        self.conv_filter = Conv1D(dilation_channels,
                                  2,
                                  dilation_rate=dilation,
                                  padding='causal',
                                  activation='tanh',
                                  use_bias=use_biases,
                                  kernel_regularizer=l2(l=regularizer),
                                  bias_regularizer=l2(l=regularizer))
        self.conv_gate = Conv1D(dilation_channels,
                                2,
                                dilation_rate=dilation,
                                padding='causal',
                                activation='sigmoid',
                                use_bias=use_biases,
                                kernel_regularizer=l2(l=regularizer),
                                bias_regularizer=l2(l=regularizer))
        self.transformed = Conv1D(residual_channels,
                                  1,
                                  padding='same',
                                  use_bias=use_biases,
                                  kernel_regularizer=l2(l=regularizer),
                                  bias_regularizer=l2(l=regularizer))
        self.skip_contribution = Conv1D(skip_channels,
                                        1,
                                        padding='same',
                                        use_bias=use_biases,
                                        kernel_regularizer=l2(l=regularizer),
                                        bias_regularizer=l2(l=regularizer))
        self.last_layer = last_layer

    def call(self, inputs, training=None):

        if self.last_layer:
            # [b, sample, residual_channels]
            filters = self.conv_filter(inputs)
            gates = self.conv_gate(inputs)
            out = filters * gates

            # The 1x1 conv to produce the skip output
            skip_cut = tf.shape(out)[1] - self.output_width
            out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])

            skip_contribution = self.skip_contribution(out_skip)

            return skip_contribution

        else:
            # [b, sample, residual_channels]
            filters = self.conv_filter(inputs)
            gates = self.conv_gate(inputs)
            out = filters * gates

            # The 1x1 conv to produce the residual output
            transformed = self.transformed(out)

            # The 1x1 conv to produce the skip output
            skip_cut = tf.shape(out)[1] - self.output_width
            out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])

            skip_contribution = self.skip_contribution(out_skip)

            return skip_contribution, inputs + transformed


class WaveNet(Model):
    '''Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
    '''
    def __init__(self, batch_size, dilations, filter_width, signal_length,
                 residual_channels, dilation_channels, skip_channels,
                 quantization_channels, use_biases, regularizer):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            signal_length: The length of input wave.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            regularizer: Regularzation weight

            TODO:
            scalar_input: Whether to use the quantized waveform directly as
                input to the network instead of one-hot encoding it.
                Default: False.
            initial_filter_width: The width of the initial filter of the
                convolution applied to the scalar input. This is only relevant
                if scalar_input=True.
            global_condition_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
            global_condition_cardinality: Number of mutually exclusive
                categories to be embedded in global condition embedding. If
                not None, then this implies that global_condition tensor
                specifies an integer selecting which of the N global condition
                categories, where N = global_condition_cardinality. If None,
                then the global_condition tensor is regarded as a vector which
                must have dimension global_condition_channels.

        '''
        super(WaveNet, self).__init__()

        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.regularizer = regularizer
        self.signal_length = signal_length
        self.receptive_field = WaveNet.calculate_receptive_field(
            self.filter_width, self.dilations)
        self.output_width = WaveNet.calculate_output_width(
            self.signal_length, self.receptive_field)
        self.pre_block = self._build_preprocess_block()
        self.residual_blocks = self._build_residual_blocks()
        self.post_block = self._build_postprocess_block()

    @staticmethod
    def calculate_receptive_field(filter_width, dilations):
        return (filter_width - 1) * sum(dilations) + filter_width

    @staticmethod
    def calculate_output_width(signal_length, receptive_field):
        return signal_length - receptive_field + 1

    def _build_preprocess_block(self):
        pre_block = Sequential()
        pre_block.add(
            Conv1D(self.residual_channels,
                   self.filter_width,
                   padding='causal',
                   use_bias=self.use_biases,
                   kernel_regularizer=l2(l=self.regularizer),
                   bias_regularizer=l2(l=self.regularizer)))
        return pre_block

    def _build_residual_blocks(self):
        outputs = []
        # Add all defined dilation layers.
        inputs = Input(shape=(self.signal_length, self.residual_channels))
        current_layer = inputs
        for i, dilation in enumerate(self.dilations):
            if i == len(self.dilations) - 1:
                output = DilatedBlock(dilation, self.output_width,
                                      self.residual_channels,
                                      self.dilation_channels,
                                      self.skip_channels, self.use_biases,
                                      self.regularizer, True)(current_layer)
            else:
                output, current_layer = DilatedBlock(
                    dilation, self.output_width, self.residual_channels,
                    self.dilation_channels, self.skip_channels,
                    self.use_biases, self.regularizer, False)(current_layer)
            outputs.append(output)

        outputs = Add()(outputs)
        return Model(inputs, outputs)

    def _build_postprocess_block(self):
        post_block = Sequential()
        post_block.add(Activation('relu'))
        post_block.add(
            Conv1D(self.skip_channels,
                   1,
                   padding='same',
                   strides=1,
                   use_bias=self.use_biases,
                   kernel_regularizer=l2(l=self.regularizer),
                   bias_regularizer=l2(l=self.regularizer),
                   activation='relu'))
        post_block.add(
            Conv1D(self.quantization_channels,
                   1,
                   padding='same',
                   strides=1,
                   use_bias=self.use_biases,
                   kernel_regularizer=l2(l=self.regularizer),
                   bias_regularizer=l2(l=self.regularizer)))
        # post_block.add(Activation('softmax'))
        return post_block

    def call(self, inputs, training=None):

        x = self.pre_block(inputs, training=training)
        x = self.residual_blocks(x, training=training)
        x = self.post_block(x, training=training)

        return x


class EEGWaveNetv1(Model):
    '''
    Implements the WaveNet network for EEG classification.
    The shape of inputs must be [batch_size, signal_length, data_channels, 1].
    Set tensorflow data format as channle last
    '''
    def __init__(self, signal_length, data_channels, dilations, filter_width,
                 residual_channels, dilation_channels, skip_channels,
                 use_biases, regularizer):
        '''
        Initializes the EEGWaveNet model.
        
        Args:
            signal_length: How long of raw data.
            data_channels: How many channels of raw data.
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            regularizer: Regularzation weight
        '''
        super(EEGWaveNetv1, self).__init__()
        self.signal_length = signal_length
        self.data_channels = data_channels
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.use_biases = use_biases
        self.regularizer = regularizer
        self.pre_block = self._build_preprocess_block()
        self.residual_blocks = self._build_residual_blocks()
        self.post_block = self._build_postprocess_block()

    def _build_preprocess_block(self):
        # [batch_size, data_channels, signal_length, 1]
        pre_block = Sequential(
            [
                Input((self.data_channels, self.signal_length, 1)),
                Conv2D(self.residual_channels, (1, self.filter_width),
                       padding='same',
                       use_bias=self.use_biases,
                       kernel_regularizer=l2(l=self.regularizer),
                       bias_regularizer=l2(l=self.regularizer)),
                BatchNormalization(),
                Activation('elu'),
                # [batch_size, data_channels, signal_length, residual_channels]
                Conv2D(self.residual_channels, (self.data_channels, 1),
                       use_bias=self.use_biases,
                       kernel_regularizer=l2(l=self.regularizer),
                       bias_regularizer=l2(l=self.regularizer)),
                BatchNormalization(),
                Activation('elu'),
                # [batch_size, 1, signal_length, residual_channels]
                Lambda(tf.squeeze, arguments=dict(axis=1))
            ],
            name='preprocess_block')
        # [batch_size, signal_length, residual_channels]
        return pre_block

    def _build_residual_blocks(self):
        def single_block(inputs):
            # [b, sample, residual_channels]
            filters = Conv1D(self.dilation_channels,
                             self.filter_width,
                             dilation_rate=dilation,
                             padding='causal',
                             activation='tanh',
                             use_bias=self.use_biases,
                             kernel_regularizer=l2(l=self.regularizer),
                             bias_regularizer=l2(l=self.regularizer))(inputs)
            gates = Conv1D(self.dilation_channels,
                           self.filter_width,
                           dilation_rate=dilation,
                           padding='causal',
                           activation='sigmoid',
                           use_bias=self.use_biases,
                           kernel_regularizer=l2(l=self.regularizer),
                           bias_regularizer=l2(l=self.regularizer))(inputs)
            out = filters * gates
            skip_contribution = Conv1D(
                self.skip_channels,
                1,
                padding='same',
                use_bias=self.use_biases,
                kernel_regularizer=l2(l=self.regularizer),
                bias_regularizer=l2(l=self.regularizer))(out)
            if last_layer:
                return skip_contribution, None
            else:
                transformed = Conv1D(
                    self.residual_channels,
                    1,
                    padding='same',
                    use_bias=self.use_biases,
                    kernel_regularizer=l2(l=self.regularizer),
                    bias_regularizer=l2(l=self.regularizer))(out)
                return skip_contribution, inputs + transformed

        outputs = []
        # Add all defined dilation layers.
        # [batch_size, signal_length, residual_channels]
        inputs = Input(shape=(self.signal_length, self.residual_channels))
        current_layer = inputs
        for i, dilation in enumerate(self.dilations):
            last_layer = (i == (len(self.dilations) - 1))
            output, current_layer = single_block(current_layer)
            outputs.append(output)

        outputs = Add()(outputs)
        # [batch_size, signal_length, skip_channels]
        return Model(inputs, outputs, name='residual_blocks')

    def _build_postprocess_block(self):
        post_block = Sequential([
            Input((self.signal_length, self.skip_channels)),
            BatchNormalization(),
            Activation('elu'),
            Dropout(0.5),
            Conv1D(16,
                   1,
                   padding='same',
                   strides=1,
                   use_bias=self.use_biases,
                   kernel_regularizer=l2(l=self.regularizer),
                   bias_regularizer=l2(l=self.regularizer)),
            BatchNormalization(),
            Activation('elu'),
            Dropout(0.5),
            AveragePooling1D(pool_size=4),
            Conv1D(8,
                   3,
                   padding='valid',
                   strides=1,
                   use_bias=self.use_biases,
                   kernel_regularizer=l2(l=self.regularizer),
                   bias_regularizer=l2(l=self.regularizer)),
            BatchNormalization(),
            Activation('elu'),
            Dropout(0.5),
            AveragePooling1D(pool_size=4),
            Conv1D(4,
                   1,
                   padding='same',
                   strides=1,
                   use_bias=self.use_biases,
                   kernel_regularizer=l2(l=self.regularizer),
                   bias_regularizer=l2(l=self.regularizer)),
            Flatten(),
            Dense(2,
                  kernel_regularizer=l2(l=self.regularizer),
                  bias_regularizer=l2(l=self.regularizer))
        ])

        return post_block

    def call(self, inputs, training=None):

        x = self.pre_block(inputs, training=training)
        x = self.residual_blocks(x, training=training)
        x = self.post_block(x, training=training)

        return x


class EEGWaveNetv2(Model):
    '''
    Implements the WaveNet network for EEG classification.
    The shape of inputs must be [batch_size, signal_length, data_channels, 1].
    Set tensorflow data format as channle last
    '''
    def __init__(self, signal_length, data_channels, dilations, filter_width,
                 residual_channels, dilation_channels, skip_channels,
                 use_biases, regularizer):
        '''
        Initializes the EEGWaveNet model.
        
        Args:
            signal_length: How long of raw data.
            data_channels: How many channels of raw data.
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            regularizer: Regularzation weight
        '''
        super(EEGWaveNetv2, self).__init__()
        self.signal_length = signal_length
        self.data_channels = data_channels
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.use_biases = use_biases
        self.regularizer = regularizer
        self.pre_block = self._build_preprocess_block()
        self.residual_blocks = self._build_residual_blocks()
        self.post_block = self._build_postprocess_block()

    def _build_preprocess_block(self):
        pre_block = Sequential(
            [  # [batch_size, data_channels, signal_length, 1]
                Input((self.data_channels, self.signal_length, 1)),
                Conv2D(self.residual_channels, (1, 1),
                       padding='same',
                       use_bias=self.use_biases,
                       kernel_regularizer=l2(l=self.regularizer),
                       bias_regularizer=l2(l=self.regularizer)),
                BatchNormalization(),
                Activation('elu'),
                # Dropout(0.2),
                # [batch_size, data_channels, signal_length, residual_channels]
            ],
            name='preprocess_block')
        # [batch_size, signal_length, residual_channels]
        return pre_block

    def _build_residual_blocks(self):
        def single_block(inputs):
            # [b, sample, residual_channels]
            filters = Conv2D(self.dilation_channels, (1, self.filter_width),
                             dilation_rate=dilation,
                             padding='same',
                             activation='tanh',
                             use_bias=self.use_biases,
                             kernel_regularizer=l2(l=self.regularizer),
                             bias_regularizer=l2(l=self.regularizer))(inputs)
            gates = Conv2D(self.dilation_channels, (1, self.filter_width),
                           dilation_rate=dilation,
                           padding='same',
                           activation='sigmoid',
                           use_bias=self.use_biases,
                           kernel_regularizer=l2(l=self.regularizer),
                           bias_regularizer=l2(l=self.regularizer))(inputs)
            out = filters * gates
            skip_contribution = Conv2D(
                self.skip_channels, (1, 1),
                padding='same',
                use_bias=self.use_biases,
                kernel_regularizer=l2(l=self.regularizer),
                bias_regularizer=l2(l=self.regularizer))(out)
            if last_layer:
                return skip_contribution, None
            else:
                transformed = Conv2D(
                    self.residual_channels, (1, 1),
                    padding='same',
                    use_bias=self.use_biases,
                    kernel_regularizer=l2(l=self.regularizer),
                    bias_regularizer=l2(l=self.regularizer))(out)
                return skip_contribution, inputs + transformed

        outputs = []
        # Add all defined dilation layers.
        # [batch_size, data_channels, signal_length, residual_channels]
        inputs = Input(shape=(self.data_channels, self.signal_length,
                              self.residual_channels))
        current_layer = inputs
        for i, dilation in enumerate(self.dilations):
            last_layer = (i == (len(self.dilations) - 1))
            output, current_layer = single_block(current_layer)
            outputs.append(output)

        outputs = Add()(outputs)
        # [batch_size, data_channels, signal_length, skip_channels * len(dilations)]
        return Model(inputs, outputs, name='residual_blocks')

    def _build_postprocess_block(self):
        post_block = Sequential([
            Input(
                (self.data_channels, self.signal_length, self.skip_channels)),
            BatchNormalization(),
            Activation('elu'),
            # Dropout(0.2),
            Conv2D(self.skip_channels * 2, (1, 1),
                   padding='same',
                   strides=1,
                   use_bias=self.use_biases,
                   kernel_regularizer=l2(l=self.regularizer),
                   bias_regularizer=l2(l=self.regularizer)),
            BatchNormalization(),
            Activation('elu'),
            # Dropout(0.2),
            AveragePooling2D(pool_size=(1, 8)),
            Conv2D(self.skip_channels * 4, (self.data_channels, 3),
                   padding='valid',
                   strides=1,
                   use_bias=self.use_biases,
                   kernel_regularizer=l2(l=self.regularizer),
                   bias_regularizer=l2(l=self.regularizer)),
            BatchNormalization(),
            Activation('elu'),
            # Dropout(0.2),
            Flatten(),
            Dense(200,
                  kernel_regularizer=l2(l=self.regularizer),
                  bias_regularizer=l2(l=self.regularizer)),
            BatchNormalization(),
            Activation('elu'),
            Dropout(0.2),
            Dense(2,
                  kernel_regularizer=l2(l=self.regularizer),
                  bias_regularizer=l2(l=self.regularizer))
        ])

        return post_block

    def call(self, inputs, training=None):
        x = self.pre_block(inputs, training=training)
        x = self.residual_blocks(x, training=training)
        x = self.post_block(x, training=training)
        return x


class EEGWaveNetv3(Model):
    '''
    Implements the WaveNet network for EEG classification.
    The shape of inputs must be [batch_size, signal_length, data_channels, 1].
    Set tensorflow data format as channle last
    '''
    def __init__(self, signal_length, data_channels, dilations, filter_width,
                 residual_channels, dilation_channels, skip_channels,
                 use_biases, regularizer):
        '''
        Initializes the EEGWaveNet model.
        
        Args:
            signal_length: How long of raw data.
            data_channels: How many channels of raw data.
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            regularizer: Regularzation weight
        '''
        super(EEGWaveNetv3, self).__init__()
        self.signal_length = signal_length
        self.data_channels = data_channels
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.use_biases = use_biases
        self.regularizer = regularizer
        self.pre_block = self._build_preprocess_block()
        self.residual_blocks = self._build_residual_blocks()
        self.post_block = self._build_postprocess_block()

    def _build_preprocess_block(self):
        pre_block = Sequential(
            [  # [batch_size, data_channels, signal_length, 1]
                Input((self.data_channels, self.signal_length, 1)),
                Conv2D(self.residual_channels, (1, 1),
                       padding='same',
                       use_bias=self.use_biases,
                       kernel_regularizer=l2(l=self.regularizer),
                       bias_regularizer=l2(l=self.regularizer)),
                BatchNormalization(),
                Activation('elu'),
                Dropout(0.2),
                # [batch_size, data_channels, signal_length, residual_channels]
            ],
            name='preprocess_block')
        # [batch_size, signal_length, residual_channels]
        return pre_block

    def _build_residual_blocks(self):
        def single_block(inputs):
            # [b, sample, residual_channels]
            filters = Conv2D(self.dilation_channels, (1, self.filter_width),
                             dilation_rate=dilation,
                             padding='same',
                             activation='tanh',
                             use_bias=self.use_biases,
                             kernel_regularizer=l2(l=self.regularizer),
                             bias_regularizer=l2(l=self.regularizer))(inputs)
            gates = Conv2D(self.dilation_channels, (1, self.filter_width),
                           dilation_rate=dilation,
                           padding='same',
                           activation='sigmoid',
                           use_bias=self.use_biases,
                           kernel_regularizer=l2(l=self.regularizer),
                           bias_regularizer=l2(l=self.regularizer))(inputs)
            out = filters * gates
            skip_contribution = Conv2D(
                self.skip_channels, (1, 1),
                padding='same',
                use_bias=self.use_biases,
                kernel_regularizer=l2(l=self.regularizer),
                bias_regularizer=l2(l=self.regularizer))(out)
            if last_layer:
                return skip_contribution, None
            else:
                transformed = Conv2D(
                    self.residual_channels, (1, 1),
                    padding='same',
                    use_bias=self.use_biases,
                    kernel_regularizer=l2(l=self.regularizer),
                    bias_regularizer=l2(l=self.regularizer))(out)
                return skip_contribution, inputs + transformed

        outputs = []
        # Add all defined dilation layers.
        # [batch_size, data_channels, signal_length, residual_channels]
        inputs = Input(shape=(self.data_channels, self.signal_length,
                              self.residual_channels))
        current_layer = inputs
        for i, dilation in enumerate(self.dilations):
            last_layer = (i == (len(self.dilations) - 1))
            output, current_layer = single_block(current_layer)
            outputs.append(output)

        outputs = Concatenate()(outputs)
        # [batch_size, data_channels, signal_length, skip_channels * len(dilations)]
        return Model(inputs, outputs, name='residual_blocks')

    def _build_postprocess_block(self):
        post_block = Sequential([
            Input((self.data_channels, self.signal_length,
                   self.skip_channels * len(self.dilations))),
            BatchNormalization(),
            Activation('elu'),
            Dropout(0.2),
            DepthwiseConv2D((self.data_channels, 1),
                            use_bias=self.use_biases,
                            depthwise_regularizer=l2(l=self.regularizer),
                            bias_regularizer=l2(l=self.regularizer)),
            BatchNormalization(),
            Activation('elu'),
            AveragePooling2D((1, 4)),
            Dropout(0.2),
            SeparableConv2D(self.skip_channels * len(self.dilations) * 2,
                            (1, 3),
                            padding='valid',
                            strides=1,
                            use_bias=self.use_biases,
                            kernel_regularizer=l2(l=self.regularizer),
                            bias_regularizer=l2(l=self.regularizer)),
            BatchNormalization(),
            Activation('elu'),
            AveragePooling2D((1, 8)),
            Dropout(0.2),
            Flatten(),
            # Dense(200,
            #       kernel_regularizer=l2(l=self.regularizer),
            #       bias_regularizer=l2(l=self.regularizer)),
            # BatchNormalization(),
            # Activation('elu'),
            # Dropout(0.2),
            Dense(2,
                  kernel_regularizer=l2(l=self.regularizer),
                  bias_regularizer=l2(l=self.regularizer))
        ])

        return post_block

    def call(self, inputs, training=None):

        x = self.pre_block(inputs, training=training)
        x = self.residual_blocks(x, training=training)
        x = self.post_block(x, training=training)

        return x


class EEGWaveNetv4(Model):
    '''
    Implements the WaveNet network for EEG classification.
    The shape of inputs must be [batch_size, signal_length, data_channels, 1].
    Set tensorflow data format as channle last
    '''
    def __init__(self, signal_length, data_channels, dilations, filter_width,
                 residual_channels, dilation_channels, skip_channels,
                 use_biases, regularizer):
        '''
        Initializes the EEGWaveNet model.
        
        Args:
            signal_length: How long of raw data.
            data_channels: How many channels of raw data.
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            regularizer: Regularzation weight
        '''
        super(EEGWaveNetv4, self).__init__()
        self.signal_length = signal_length
        self.data_channels = data_channels
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.use_biases = use_biases
        self.regularizer = regularizer
        self.pre_block = self._build_preprocess_block()
        self.residual_blocks = self._build_residual_blocks()
        self.post_block = self._build_postprocess_block()

    def _build_preprocess_block(self):
        pre_block = Sequential(
            [  # [batch_size, data_channels, signal_length, 1]
                Input((self.data_channels, self.signal_length, 1)),
                Conv2D(self.residual_channels, (1, 1),
                       padding='same',
                       use_bias=self.use_biases,
                       kernel_regularizer=l2(l=self.regularizer),
                       bias_regularizer=l2(l=self.regularizer)),
                BatchNormalization(),
                Activation('elu'),
                Dropout(0.5),
                # [batch_size, data_channels, signal_length, residual_channels]
            ],
            name='preprocess_block')
        # [batch_size, signal_length, residual_channels]
        return pre_block

    def _build_residual_blocks(self):
        def single_block(inputs):
            # [b, sample, residual_channels]
            filters = Conv2D(self.dilation_channels, (1, self.filter_width),
                             dilation_rate=dilation,
                             padding='same',
                             activation='tanh',
                             use_bias=self.use_biases,
                             kernel_regularizer=l2(l=self.regularizer),
                             bias_regularizer=l2(l=self.regularizer))(inputs)
            gates = Conv2D(self.dilation_channels, (1, self.filter_width),
                           dilation_rate=dilation,
                           padding='same',
                           activation='sigmoid',
                           use_bias=self.use_biases,
                           kernel_regularizer=l2(l=self.regularizer),
                           bias_regularizer=l2(l=self.regularizer))(inputs)
            out = filters * gates
            skip_contribution = Conv2D(
                self.skip_channels, (1, 1),
                padding='same',
                use_bias=self.use_biases,
                kernel_regularizer=l2(l=self.regularizer),
                bias_regularizer=l2(l=self.regularizer))(out)
            if last_layer:
                return skip_contribution, None
            else:
                transformed = Conv2D(
                    self.residual_channels, (1, 1),
                    padding='same',
                    use_bias=self.use_biases,
                    kernel_regularizer=l2(l=self.regularizer),
                    bias_regularizer=l2(l=self.regularizer))(out)
                return skip_contribution, inputs + transformed

        outputs = []
        # Add all defined dilation layers.
        # [batch_size, data_channels, signal_length, residual_channels]
        inputs = Input(shape=(self.data_channels, self.signal_length,
                              self.residual_channels))
        current_layer = inputs
        for i, dilation in enumerate(self.dilations):
            last_layer = (i == (len(self.dilations) - 1))
            output, current_layer = single_block(current_layer)
            outputs.append(output)

        outputs = Concatenate()(outputs)
        # [batch_size, data_channels, signal_length, skip_channels * len(dilations)]
        return Model(inputs, outputs, name='residual_blocks')

    def _build_postprocess_block(self):
        post_block = Sequential([
            Input((self.data_channels, self.signal_length,
                   self.skip_channels * len(self.dilations))),
            BatchNormalization(),
            Activation('elu'),
            Dropout(0.5),
            DepthwiseConv2D((self.data_channels, 1),
                            use_bias=self.use_biases,
                            depthwise_regularizer=l2(l=self.regularizer),
                            bias_regularizer=l2(l=self.regularizer)),
            BatchNormalization(),
            Activation('elu'),
            AveragePooling2D((1, 4)),
            Dropout(0.5),
            SeparableConv2D(self.skip_channels * len(self.dilations) * 2,
                            (1, 7),
                            padding='valid',
                            strides=1,
                            use_bias=self.use_biases,
                            kernel_regularizer=l2(l=self.regularizer),
                            bias_regularizer=l2(l=self.regularizer)),
            BatchNormalization(),
            Activation('elu'),
            AveragePooling2D((1, 8)),
            Dropout(0.5),
            Flatten(),
            # Dense(200,
            #       kernel_regularizer=l2(l=self.regularizer),
            #       bias_regularizer=l2(l=self.regularizer)),
            # BatchNormalization(),
            # Activation('elu'),
            # Dropout(0.2),
            Dense(2,
                  kernel_regularizer=l2(l=self.regularizer),
                  bias_regularizer=l2(l=self.regularizer))
        ])

        return post_block

    def call(self, inputs, training=None):

        x = self.pre_block(inputs, training=training)
        x = self.residual_blocks(x, training=training)
        x = self.post_block(x, training=training)

        return x


class EEGWaveNetv5(Model):
    '''
    Implements the WaveNet network for EEG classification.
    The shape of inputs must be [batch_size, signal_length, data_channels, 1].
    Set tensorflow data format as channle last
    '''
    def __init__(self,
                 nclass,
                 signal_length,
                 data_channels,
                 kernel_size,
                 sample_rate,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 min_low_hz=1,
                 min_band_hz=1,
                 dropout_rate=0,
                 use_biases=False,
                 regularizer=0,
                 *args,
                 **kwargs):
        '''
        Initializes the EEGWaveNet model.
        
        Args:
            nclass
            signal_length: How long of raw data.
            data_channels: How many channels of raw data.
            kernel_size
            sample_rate
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            min_low_hz
            min_band_hz
            dropout_rate
            use_biases: Whether to add a bias layer to each convolution.
                Default: False.
            regularizer: Regularzation weight
        '''
        super(EEGWaveNetv5, self).__init__()
        self.nclass = nclass
        self.signal_length = signal_length
        self.data_channels = data_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.dropout_rate = dropout_rate
        self.use_biases = use_biases
        self.regularizer = regularizer

        self.pre_block = self._build_preprocess_block()
        self.residual_blocks = self._build_residual_blocks()
        self.post_block = self._build_postprocess_block()

    def _build_preprocess_block(self):
        pre_block = Sequential(
            [  # [batch_size, data_channels, signal_length, 1]
                Input((self.data_channels, self.signal_length, 1)),
                SincConv(filters=self.residual_channels,
                         kernel_size=self.kernel_size,
                         sample_rate=self.sample_rate,
                         min_low_hz=self.min_low_hz,
                         min_band_hz=self.min_band_hz),
                BatchNormalization(momentum=0.85),
                DepthwiseConv2D((self.data_channels, 1),
                                use_bias=self.use_biases,
                                depthwise_regularizer=l2(self.regularizer)),
                BatchNormalization(momentum=0.85),
                Dropout(self.dropout_rate),
            ],
            name='preprocess_block')
        # [batch_size, 1, signal_length, residual_channels]
        return pre_block

    def _build_residual_blocks(self):
        def single_block(inputs):
            # [b, 1, sample, residual_channels]
            filters = Conv2D(self.dilation_channels, (1, self.filter_width),
                             dilation_rate=dilation,
                             padding='same',
                             activation='tanh',
                             use_bias=self.use_biases,
                             kernel_regularizer=l2(self.regularizer),
                             bias_regularizer=l2(self.regularizer))(inputs)
            gates = Conv2D(self.dilation_channels, (1, self.filter_width),
                           dilation_rate=dilation,
                           padding='same',
                           activation='sigmoid',
                           use_bias=self.use_biases,
                           kernel_regularizer=l2(self.regularizer),
                           bias_regularizer=l2(self.regularizer))(inputs)
            out = filters * gates
            skip_contribution = Conv2D(self.skip_channels, (1, 1),
                                       padding='same',
                                       use_bias=self.use_biases,
                                       kernel_regularizer=l2(self.regularizer),
                                       bias_regularizer=l2(
                                           self.regularizer))(out)
            if last_layer:
                return skip_contribution, None
            else:
                transformed = Conv2D(self.residual_channels, (1, 1),
                                     padding='same',
                                     use_bias=self.use_biases,
                                     kernel_regularizer=l2(self.regularizer),
                                     bias_regularizer=l2(
                                         self.regularizer))(out)
                return skip_contribution, inputs + transformed

        outputs = []
        # Add all defined dilation layers.
        # [batch_size, 1, signal_length, residual_channels]
        inputs = Input(shape=(1, self.signal_length, self.residual_channels))
        current_layer = inputs
        for i, dilation in enumerate(self.dilations):
            last_layer = (i == (len(self.dilations) - 1))
            output, current_layer = single_block(current_layer)
            outputs.append(output)

        outputs = Concatenate()(outputs)
        # [batch_size, 1, signal_length, skip_channels * len(dilations)]
        return Model(inputs, outputs, name='residual_blocks')

    def _build_postprocess_block(self):
        post_block = Sequential([
            Input((1, self.signal_length,
                   self.skip_channels * len(self.dilations))),
            BatchNormalization(momentum=0.85),
            Activation('relu'),
            AveragePooling2D((1, 4)),
            Dropout(self.dropout_rate),
            DepthwiseConv2D((1, 16), use_bias=self.use_biases, padding='same'),
            BatchNormalization(momentum=0.85),
            Activation('relu'),
            Dropout(self.dropout_rate),
            Conv2D(self.skip_channels * len(self.dilations),
                   1,
                   use_bias=self.use_biases),
            BatchNormalization(momentum=0.85),
            Activation('relu'),
            AveragePooling2D((1, 8)),
            Dropout(self.dropout_rate),
            Flatten(),
            Dense(self.nclass),
            Activation('softmax', name='softmax')
        ])
        return post_block

    def call(self, inputs, training=None):
        x = self.pre_block(inputs, training=training)
        x = self.residual_blocks(x, training=training)
        x = self.post_block(x, training=training)
        return x
