import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Conv1D, Input, Add, Activation
from tensorflow.keras.regularizers import l2


class DilatedBlock(Layer):
    """
    Creates a single causal dilated convolution layer

                |-> [gate]   -|        |-> 1x1 conv -> skip output
                |             |-> (*) -|
         input -|-> [filter] -|        |-> 1x1 conv -|
                |                                    |-> (+) -> dense output
                |------------------------------------|
    """
    def __init__(self, dilation, output_width, residual_channels,
                 dilation_channels, skip_channels, use_biases, regularizer):
        super(DilatedBlock, self).__init__()

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

    def call(self, inputs, training=None):
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
    def __init__(self, dilations, filter_width, signal_length,
                 residual_channels, dilation_channels, skip_channels,
                 quantization_channels, use_biases, regularizer):
        '''Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
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
            scalar_input: Whether to use the quantized waveform directly as
                input to the network instead of one-hot encoding it.
                Default: False.
            initial_filter_width: The width of the initial filter of the
                convolution applied to the scalar input. This is only relevant
                if scalar_input=True.
            histograms: Whether to store histograms in the summary.
                Default: False.
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
        for dilation in self.dilations:
            output, current_layer = DilatedBlock(
                dilation, self.output_width, self.residual_channels,
                self.dilation_channels, self.skip_channels, self.use_biases,
                self.regularizer)(current_layer)
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
