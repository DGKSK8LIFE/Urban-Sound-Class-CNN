# I DID NOT WRITE THE ORIGINAL CODE
# I EDITED IT A BIT, BUT THE REAL AUTHOR IS BELOW
# http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
# Just a great visual way to show the audio
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

# Load the sound files with librosa
def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X, sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

# Regular waveform (Amplitude vs time)
def plot_waves(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(5, 12), dpi=100)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot", x=0.5, y=0.915, fontsize=10)
    plt.show()

# Spectrogram (amplitude vs Hz)
def plot_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(5, 12), dpi=100)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectrogram", x=0.5, y=0.915, fontsize=10)
    plt.show()

# Log power spectrogram (amplitude vs Hz)
def plot_log_power_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(5, 12), dpi=100)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(10, 1, i)
        D = librosa.core.amplitude_to_db(np.abs(librosa.stft(f)) ** 2, ref=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 3: Log power spectrogram", x=0.5, y=0.915, fontsize=10)
    plt.show()

test_audio = 'testaudio/'

sound_files = ["57320-0-0-7.wav", "24074-1-0-3.wav", "15564-2-0-1.wav", "31323-3-0-1.wav", "46669-4-0-35.wav",
                    "89948-5-0-0.wav", "40722-8-0-4.wav", "103074-7-3-2.wav", "106905-8-0-0.wav", "108041-9-0-4.wav"]
sound_file_paths = []
for i in sound_files:
    test_audio_paths = test_audio + i
    sound_file_paths.append(test_audio_paths)

sound_names = ["air conditioner", "car horn", "children playing", "dog bark", "drilling", "engine idling", "gun shot",
               "jackhammer", "siren", "street music"]

raw_sounds = load_sound_files(sound_file_paths)
plot_waves(sound_names, raw_sounds)
plot_specgram(sound_names, raw_sounds)
plot_log_power_specgram(sound_names, raw_sounds)


