import os
import numpy as np
import random
import itertools
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt

def load_wave_data(file_name):
    # file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_name, sr=44100)
    return x,fs

# change wave data to mel-stft
def calculate_melsp(x, sr):
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=2048, win_length=512, hop_length=512)
    log_stft = librosa.power_to_db(S, ref=np.max)
    return log_stft

# display wave in plots
def show_wave(x):
    plt.plot(x)
    plt.savefig("wave.png")
    plt.show()
    plt.close()

# display wave in heatmap
def show_melsp(melsp, sr):
    librosa.display.specshow(melsp, sr=sr)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig("melsp.png")
    plt.show()
    plt.close()


def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))

# data augmentation: shift sound in timeframe
def shift_sound(x, rate=2):
    return np.roll(x, int(len(x)//rate))

# data augmentation: stretch sound
def stretch_sound(x, rate=1.1):
    input_length = len(x)
    x = librosa.effects.time_stretch(x, rate)
    if len(x)>input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length - len(x))), "constant")


x, sr = load_wave_data("audio/1-137-A-32.wav")
# show_wave(x)
melsp = calculate_melsp(x, sr)
print("wave size:{0}\nmelsp size:{1}\nsamping rate:{2}".format(x.shape, melsp.shape, sr))
show_melsp(melsp, sr)

