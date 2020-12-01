import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from deepeeg.utils import hilbert


def test():
    duration = 1.0
    fs = 400.0
    samples = int(fs*duration)
    t = np.arange(samples) / fs

    signal = chirp(t, 20.0, t[-1], 100.0)
    signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )
    signal = np.tile(signal, (9, 1))
    signal = np.reshape(signal, [3, 1, 3, -1])
    signal = np.transpose(signal, [0, 1, 3, 2])

    analytic_signal = hilbert(signal)

    amplitude_envelope = np.abs(analytic_signal)

    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax0.plot(t, signal[0, 0, :, 0], label='signal')
    ax0.plot(t, amplitude_envelope[0, 0, :, 0], label='envelope')
    ax0.set_xlabel("time in seconds")
    ax0.legend()
    plt.show()


if __name__ == "__main__":
    test()