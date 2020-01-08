# Extremely helpful link I used
# http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/

import pandas as pd
# Used for data processing and feature extraction. Package for audio analysis
import librosa
import numpy as np
# tqdm shows a progress meter
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from os import path, chdir

parent = 'ttdl/'

# CSV file with all the information on the sound files including labels, file names, folder, etc.
# /home/galloj/anaconda3/envs/aiml/bin/python -->  /home/galloj/Documents/aiml/Data/UrbanSound8K/metadata/UrbanSound8K.csv
# They have /home/galloj/ in common
# ../../../../ is for going back thru /envs/aiml/bin/python/ and reaching the common directory
data = pd.read_csv('../../../../Documents/aiml/Data/UrbanSound8K/metadata/UrbanSound8K.csv')


# Function for checking if the train and test CSV files exists
def path_exists():
    if path.exists(parent + 'test_data.csv') and path.exists(parent + 'test_labels.csv') and \
            path.exists(parent + 'train_data.csv') and path.exists(parent + 'train_labels.csv'):
        return True
    else:
        return False


# Check if the test and train CSV files exists, so the program doesn't extract the features every time it runs (they are
# already saved in the CSV files)
if not path_exists():

    # The train accuracy: The accuracy of a model on examples it was constructed on (fold 1 - 9)
    # The test accuracy is the accuracy of a model on examples it hasn't seen (fold 10)
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    sound_path = "UrbanSound8K/audio/fold"

    # Go through the whole 494 kB CSV file with the progress meter
    for i in tqdm(range(len(data))):

        # Retrieve the fold, slice_file_name_ and classID rows of each audio file in the CSV file
        fold_no = str(data.iloc[i]["fold"])
        file = data.iloc[i]["slice_file_name"]
        label = data.iloc[i]["classID"]

        # The directory for each specific .wav file
        filename = sound_path + fold_no + "/" + file

        # Load each audio file as a floating point time series
        y, sr = librosa.load(filename)

        # Processing using entire feature set including:
        # MFCC: Mel-frequency cepstral coefficients that use a quasi-logarithmic spaced frequency scale, which is more
        # similar to how the human auditory system processes sounds.
        # Melspectrogram: Compute a Mel-scaled power spectrogram. Based on human ear.
        # chorma-stft: Compute a chromagram from a waveform or power spectrogram. Uses pitches.
        # chroma_cq: Constant-Q chromogram. Uses pitches.
        # chroma_cens: Chroma Energy Normalized CENS. Uses pitches.

        # Return 40 samples of each feature per sound file
        mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T, axis=0)
        melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
        chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40).T, axis=0)
        chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40).T, axis=0)

        # Compiling all the features into one array with shape (40, 5)
        features = np.reshape(np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)), (40, 5))

        # Training the model on folders 1 - 9, testing on fold 10
        if fold_no != '10':
            x_train.append(features)
            y_train.append(label)
        else:
            x_test.append(features)
            y_test.append(label)

    # 8732 audio files
    print(len(x_train) + len(x_test))
    print(len(data))

    # Converting the lists into numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Reshaping into 2D to save in CSV format
    x_train_2d = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test_2d = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    print(x_train_2d.shape, x_test_2d.shape)

    # Saving the data numpy arrays to make it easier, only one time when the files already exist
    np.savetxt(parent + "train_data.csv", x_train_2d, delimiter=",")
    np.savetxt(parent + "test_data.csv", x_test_2d, delimiter=",")
    np.savetxt(parent + "train_labels.csv", y_train, delimiter=",")
    np.savetxt(parent + "test_labels.csv", y_test, delimiter=",")


# Extracting data from csv files into numpy arrays
x_train = np.genfromtxt(parent + 'train_data.csv', delimiter=',')
y_train = np.genfromtxt(parent + 'train_labels.csv', delimiter=',')
x_test = np.genfromtxt(parent + 'test_data.csv', delimiter=',')
y_test = np.genfromtxt(parent + 'test_labels.csv', delimiter=',')
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# One hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(y_train.shape, y_test.shape)

# Reshaping to 2D
x_train = np.reshape(x_train, (x_train.shape[0], 40, 5))
x_test = np.reshape(x_test, (x_test.shape[0], 40, 5))
print(x_train.shape, x_test.shape)

# Reshaping to shape required by CNN
x_train = np.reshape(x_train, (x_train.shape[0], 40, 5, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 40, 5, 1))
print(x_train.shape, x_test.shape)

# Creating the model using Keras, can adapt easily to Tensorflow
# Forming model
model = Sequential()

# Adding layers and forming the model
model.add(Conv2D(64, kernel_size=5, strides=1, padding="Same", activation="relu", input_shape=(40, 5, 1)))
model.add(MaxPooling2D(padding="same"))

model.add(Conv2D(128, kernel_size=5, strides=1, padding="same", activation="relu"))
model.add(MaxPooling2D(padding="same"))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(512, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(10, activation="softmax"))

# Compiling the model (Categorical cross entropy = softmax + cross entropy)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Training the model
model.fit(x_train, y_train, batch_size=50, epochs=30, validation_data=(x_test, y_test))

# Evaluate the train and test
train_loss_score = model.evaluate(x_train, y_train)
test_loss_score = model.evaluate(x_test, y_test)
print(train_loss_score)
print(test_loss_score)

# [Train loss, train accuracy]
# [Test loss, test accuracy]
