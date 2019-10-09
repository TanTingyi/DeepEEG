import os
import tensorflow as tf
import numpy as np
from wavenet import WaveNet, DilatedBlock
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

SAMPLE_RATE_HZ = 2000.0  # Hz
TRAIN_ITERATIONS = 400
SAMPLE_DURATION = 0.5  # Seconds
SAMPLE_PERIOD_SECS = 1.0 / SAMPLE_RATE_HZ
MOMENTUM = 0.95
GENERATE_SAMPLES = 1000
QUANTIZATION_CHANNELS = 256
NUM_SPEAKERS = 3
F1 = 155.56  # E-flat frequency in hz
F2 = 196.00  # G frequency in hz
F3 = 233.08  # B-flat frequency in hz


def make_sine_waves(global_conditioning):
    """Creates a time-series of sinusoidal audio amplitudes."""
    sample_period = 1.0/SAMPLE_RATE_HZ
    times = np.arange(0.0, SAMPLE_DURATION, sample_period)

    if global_conditioning:
        LEADING_SILENCE = random.randint(10, 128)
        amplitudes = np.zeros(shape=(NUM_SPEAKERS, len(times)))
        amplitudes[0, 0:LEADING_SILENCE] = 0.0
        amplitudes[1, 0:LEADING_SILENCE] = 0.0
        amplitudes[2, 0:LEADING_SILENCE] = 0.0
        start_time = LEADING_SILENCE / SAMPLE_RATE_HZ
        times = times[LEADING_SILENCE:] - start_time
        amplitudes[0, LEADING_SILENCE:] = 0.6 * np.sin(times *
                                                       2.0 * np.pi * F1)
        amplitudes[1, LEADING_SILENCE:] = 0.5 * np.sin(times *
                                                       2.0 * np.pi * F2)
        amplitudes[2, LEADING_SILENCE:] = 0.4 * np.sin(times *
                                                       2.0 * np.pi * F3)
        speaker_ids = np.zeros((NUM_SPEAKERS, 1), dtype=np.int)
        speaker_ids[0, 0] = 0
        speaker_ids[1, 0] = 1
        speaker_ids[2, 0] = 2
    else:
        amplitudes = (np.sin(times * 2.0 * np.pi * F1) / 3.0 +
                      np.sin(times * 2.0 * np.pi * F2) / 3.0 +
                      np.sin(times * 2.0 * np.pi * F3) / 3.0)
        speaker_ids = None

    return amplitudes, speaker_ids


def generate_waveform(sess, net, fast_generation, gc, samples_placeholder,
                      gc_placeholder, operations):
    waveform = [128] * net.receptive_field
    if fast_generation:
        for sample in waveform[:-1]:
            sess.run(operations, feed_dict={samples_placeholder: [sample]})

    for i in range(GENERATE_SAMPLES):
        if i % 100 == 0:
            print("Generating {} of {}.".format(i, GENERATE_SAMPLES))
            sys.stdout.flush()
        if fast_generation:
            window = waveform[-1]
        else:
            if len(waveform) > net.receptive_field:
                window = waveform[-net.receptive_field:]
            else:
                window = waveform

        # Run the WaveNet to predict the next sample.
        feed_dict = {samples_placeholder: window}
        if gc is not None:
            feed_dict[gc_placeholder] = gc
        results = sess.run(operations, feed_dict=feed_dict)

        sample = np.random.choice(
           np.arange(QUANTIZATION_CHANNELS), p=results[0])
        waveform.append(sample)

    # Skip the first number of samples equal to the size of the receptive
    # field minus one.
    waveform = np.array(waveform[net.receptive_field - 1:])
    decode = mu_law_decode(samples_placeholder, QUANTIZATION_CHANNELS)
    decoded_waveform = sess.run(decode,
                                feed_dict={samples_placeholder: waveform})
    return decoded_waveform


def generate_waveforms(sess, net, fast_generation, global_condition):
    samples_placeholder = tf.placeholder(tf.int32)
    gc_placeholder = tf.placeholder(tf.int32) if global_condition is not None \
        else None

    net.batch_size = 1

    if fast_generation:
        next_sample_probs = net.predict_proba_incremental(samples_placeholder,
                                                          global_condition)
        sess.run(net.init_ops)
        operations = [next_sample_probs]
        operations.extend(net.push_ops)
    else:
        next_sample_probs = net.predict_proba(samples_placeholder,
                                              gc_placeholder)
        operations = [next_sample_probs]

    num_waveforms = 1 if global_condition is None else  \
        global_condition.shape[0]
    gc = None
    waveforms = [None] * num_waveforms
    for waveform_index in range(num_waveforms):
        if global_condition is not None:
            gc = global_condition[waveform_index, :]
        # Generate a waveform for each speaker id.
        print("Generating waveform {}.".format(waveform_index))
        waveforms[waveform_index] = generate_waveform(
            sess, net, fast_generation, gc, samples_placeholder,
            gc_placeholder, operations)

    return waveforms, global_condition


def find_nearest(freqs, power_spectrum, frequency):
    # Return the power of the bin nearest to the target frequency.
    index = (np.abs(freqs - frequency)).argmin()
    return power_spectrum[index]


def check_waveform(assertion, generated_waveform, gc_category):
    # librosa.output.write_wav('/tmp/sine_test{}.wav'.format(gc_category),
    #                          generated_waveform,
    #                          SAMPLE_RATE_HZ)
    power_spectrum = np.abs(np.fft.fft(generated_waveform))**2
    freqs = np.fft.fftfreq(generated_waveform.size, SAMPLE_PERIOD_SECS)
    indices = np.argsort(freqs)
    indices = [index for index in indices if freqs[index] >= 0 and
               freqs[index] <= 500.0]
    power_spectrum = power_spectrum[indices]
    freqs = freqs[indices]
    # plt.plot(freqs[indices], power_spectrum[indices])
    # plt.show()
    power_sum = np.sum(power_spectrum)
    f1_power = find_nearest(freqs, power_spectrum, F1)
    f2_power = find_nearest(freqs, power_spectrum, F2)
    f3_power = find_nearest(freqs, power_spectrum, F3)
    if gc_category is None:
        # We are not globally conditioning to select one of the three sine
        # waves, so expect it across all three.
        expected_power = f1_power + f2_power + f3_power
        assertion(expected_power, 0.7 * power_sum)
    else:
        # We expect spectral power at the selected frequency
        # corresponding to the gc_category to be much higher than at the other
        # two frequencies.
        frequency_lut = {0: f1_power, 1: f2_power, 2: f3_power}
        other_freqs_lut = {0: f2_power + f3_power,
                           1: f1_power + f3_power,
                           2: f1_power + f2_power}
        expected_power = frequency_lut[gc_category]
        # Power at the selected frequency should be at least 10 times greater
        # than at other frequences.
        # This is a weak criterion, but still detects implementation errors
        # in the code.
        assertion(expected_power, 10.0*other_freqs_lut[gc_category])


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    audio = tf.cast(audio, dtype=tf.float32)
    mu = tf.cast(quantization_channels - 1, dtype=tf.float32)
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = tf.cast(tf.minimum(tf.abs(audio), 1.0), dtype=tf.float32)
    magnitude = tf.math.log1p(mu * safe_audio_abs) / tf.math.log1p(mu)
    signal = tf.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return tf.cast((signal + 1) / 2 * mu + 0.5, dtype=tf.int32)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    mu = quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * (tf.cast(output, dtype=tf.float32) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    return tf.sign(signal) * magnitude


def one_hot(input_batch, quantization_channels):
    '''One-hot encodes the waveform amplitudes.

    This allows the definition of the network as a categorical distribution
    over a finite set of possible amplitudes.
    '''
    # input: [b, smaple, channel]
    encoded = tf.one_hot(input_batch,
                         depth=quantization_channels,
                         dtype=tf.float32)
    batch_size = tf.shape(input_batch)[0]
    # batch_size = self.batch_size
    shape = [batch_size, -1, quantization_channels]
    encoded = tf.reshape(encoded, shape)
    return encoded


def test():
    np.random.seed(42)
    audio, speaker_ids = make_sine_waves(None)
    dilations = [2**i for i in range(7)] * 2
    receptive_field = WaveNet.calculate_receptive_field(2, dilations)
    audio = np.pad(audio, (receptive_field - 1, 0),
                   'constant').astype(np.float32)

    encoded = mu_law_encode(audio, 2**8)
    encoded = encoded[np.newaxis, :]
    encoded_one_hot = one_hot(encoded, 2**8)

    signal_length = int(tf.shape(encoded_one_hot)[1] - 1)
    input_one_hot = tf.slice(encoded_one_hot, [0, 0, 0],
                             [-1, signal_length, -1])
    target_one_hot = tf.slice(encoded_one_hot, [0, receptive_field, 0],
                              [-1, -1, -1])
    print('input shape: ', tf.shape(input_one_hot))
    print('output shape: ', tf.shape(target_one_hot))

    net = WaveNet(dilations, 2, signal_length, 32, 32, 32, 2**8, True, 0.01)
    net.build(input_shape=(None, signal_length, 2**8))
    optimizer = Adam(lr=1e-3)
    
    for epoch in range(301):
        with tf.GradientTape() as tape:
            # [b, 1254, 256] => [b, 999, 256]
            logits = net(input_one_hot, training=True)
            # [b, 999, 256] => [b * 999, 256]
            logits = tf.reshape(logits, [-1, 2**8])
            target_one_hot = tf.reshape(target_one_hot, [-1, 2**8])
            # comput loss
            loss = tf.losses.categorical_crossentropy(target_one_hot,
                                                      logits,
                                                      from_logits=True)
            loss = tf.reduce_mean(loss)
    
        grads = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        if epoch % 100 == 0:
            print(epoch, 'loss: ', float(loss))


if __name__ == "__main__":
    test()