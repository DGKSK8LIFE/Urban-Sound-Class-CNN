# Great way to display each audio feature used in the CNN
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Feature test and visualization on a dog bark

# This file is of a dog bark
y, sr = librosa.load("../../../../Documents/aiml/Data/UrbanSound8K/audio/fold5/100032-3-0-0.wav")

# Extracting each feature
mfccs = librosa.feature.mfcc(y, sr, n_mfcc=40)
melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40)
chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40)
chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40)
print('Shape of each individual feature:', melspectrogram.shape, chroma_stft.shape, chroma_cq.shape, chroma_cens.shape, mfccs.shape)

# MFCC: Mel-frequency cepstral coefficients that use a quasi-logarithmic spaced frequency scale, which is more similar
# to how the human auditory system processes sounds.
# Melspectrogram: Compute a Mel-scaled power spectrogram. Based on human ear.
# chorma-stft: Compute a chromagram from a waveform or power spectrogram. Uses pitches.
# chroma_cq: Constant-Q chromogram. Uses pitches.
# chroma_cens: Chroma Energy Normalized CENS. Uses pitches.

# MFCC of dog bark
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

# Melspectrogram of a dog bark
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

# Chromagram of dog bark
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()

# Chroma cqt of a dog bark
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('chroma_cqt')
plt.tight_layout()
plt.show()

# Chroma cens of a dog bark
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('chroma_cens')
plt.tight_layout()
plt.show()


