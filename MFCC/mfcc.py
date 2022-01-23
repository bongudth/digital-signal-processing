import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy.signal import find_peaks
import librosa
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

N_MFCC = 39
FRAME_TIME = 30e-3
FRAME_SHIFT_TIME = 10e-3
THRESHOLD_VOWEL_SILENCE_BY_ENERGY = 8.5e-3
N_CLUSTERS = 3

def get_frames(signal, fs):
	signal_sample = len(signal)
	frame_sample = int(FRAME_TIME * fs)
	frame_shift_sample = int(FRAME_SHIFT_TIME * fs)
	left, right = 0, frame_sample
	frames = []
	while right < signal_sample:
		frames.append(signal[left:right])
		left += frame_shift_sample
		right += frame_shift_sample
	return np.array(frames)

def energy(x):
	return np.sum(x * x)

def get_frame_vowel(signal, fs):
	frames = get_frames(signal, fs)
	max_energy = 0
	for frame in frames:
		max_energy = max(max_energy, energy(frame))
	frame_vowel = []
	for frame in frames:
		if energy(frame) >= max_energy * THRESHOLD_VOWEL_SILENCE_BY_ENERGY:
			frame_vowel.append(frame)
	n = len(frame_vowel)
	return frame_vowel[n // 3 : n // 3 * 2]

def plot_confusion_matrix(y_true, y_pred, labels):
    matrix = confusion_matrix(y_true, y_pred)    
    fig, ax = plt.subplots(figsize=(8,8))
    plt.imshow(matrix, cmap=plt.cm.Blues)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i, true_label in enumerate(matrix):
        for j, predicted_label in enumerate(true_label):
            text = ax.text(j, i, matrix[i, j], ha="center", va="center", fontweight='bold');
    plt.title("Confusion Matrix")

# KQTG - STE
# def plot_frame_vowel(signal, fs):
#     frames = get_frames(signal, fs)
#     max_energy = 0
#     for frame in frames:
#         max_energy = max(max_energy, energy(frame))
#     frame_vowel_index = []
#     for i in range(len(frames)):
#         if energy(frames[i]) >= max_energy * THRESHOLD_VOWEL_SILENCE_BY_ENERGY:
#             frame_vowel_index.append(i)
#     n = len(signal)
#     m = len(frame_vowel_index)
#     t = 1 / fs * np.arange(n)
#     plt.plot(t, signal)
#     plt.xlabel('Time(s)')
#     plt.ylabel('Amplitude')
#     plt.axvline(x=frame_vowel_index[0] * FRAME_SHIFT_TIME, color='g', linestyle='-')
#     plt.axvline(x=frame_vowel_index[-1] * FRAME_SHIFT_TIME, color='g', linestyle='-')
#     plt.axvline(x=frame_vowel_index[m // 3] * FRAME_SHIFT_TIME, color='r', linestyle='-')
#     plt.axvline(x=frame_vowel_index[m // 3 * 2] * FRAME_SHIFT_TIME, color='r', linestyle='-')

# testing_folder = './NguyenAmKiemThu-16k'
# folder = '23MTL'
# vowel = 'u'
# signal, fs = sf.read(testing_folder + '/' + folder + '/' + vowel + '.wav')
# plt.title(folder + ' - ' + vowel)
# plot_frame_vowel(signal, fs)

#KQTG_MFCC

# N_MFCC = 13
# N_CLUSTERS = 1

# vowels = ['a', 'e', 'i', 'o', 'u']
# training_folder = './NguyenAmHuanLuyen-16k' 
# training_folders = []
# for folder in os.listdir(training_folder):
#     training_folders.append(folder)

# vowels_feature = {}

# for vowel in vowels:
# 	features = []

# 	for folder in training_folders:
# 		signal, fs = sf.read(training_folder + '/' + folder + '/' + vowel + '.wav')
# 		signal = signal / np.max(signal)
# 		frames = get_frame_vowel(signal, fs)
# 		mfccs = []
# 		for frame in frames:
# 			S = librosa.feature.melspectrogram(y=frame, sr=fs)
# 			mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=N_MFCC)
# 			mfcc = mfcc.T[0]
# 			mfccs.append(mfcc)

# 		feature = np.mean(mfccs, axis=0)
# 		features.append(feature)

# 	kmeans = KMeans(n_clusters=N_CLUSTERS).fit(features)
# 	clusters = kmeans.cluster_centers_
# 	for v in clusters:
# 		plt.plot(np.arange(N_MFCC), v, label=vowel)
# 	vowels_feature[vowel] = clusters
# plt.legend()
# plt.title('N_MFCC = 13')

# TRAIN
vowels = ['a', 'e', 'i', 'o', 'u']
training_folder = './NguyenAmHuanLuyen-16k' 
training_folders = []
for folder in os.listdir(training_folder):
	training_folders.append(folder)

vowels_feature = {}

for vowel in vowels:
	features = []
	
	for folder in training_folders:
		signal, fs = sf.read(training_folder + '/' + folder + '/' + vowel + '.wav')
		signal = signal / np.max(signal)
		frames = get_frame_vowel(signal, fs)
		mfccs = []
		for frame in frames:
			S = librosa.feature.melspectrogram(y=frame, sr=fs)
			mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=N_MFCC)
			mfcc = mfcc.T[0]
			mfccs.append(mfcc)

		feature = np.mean(mfccs, axis=0)
		plt.figure(1)

		if vowel == 'a':
			plt.plot(np.arange(N_MFCC), feature)

		features.append(feature)

	kmeans = KMeans(n_clusters=N_CLUSTERS).fit(features)
	clusters = kmeans.cluster_centers_
	if vowel == 'a':
		plt.figure(2)
		for i in clusters:
			plt.plot(np.arange(N_MFCC), i)
	vowels_feature[vowel] = clusters



# TEST

vowels = ['a', 'e', 'i', 'o', 'u']
testing_folder = './NguyenAmKiemThu-16k' 
testing_folders = []
for folder in os.listdir(testing_folder):
	testing_folders.append(folder)

y_true = []
y_pred = []
ok = 0

for vowel in vowels:
	
	for folder in testing_folders:
		signal, fs = sf.read(testing_folder + '/' + folder + '/' + vowel + '.wav')
		signal = signal / np.max(signal)
		frames = get_frame_vowel(signal, fs)
		mfccs = []
		for frame in frames:
			S = librosa.feature.melspectrogram(y=frame, sr=fs)
			mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=N_MFCC)
			mfcc = mfcc.T[0]
			mfccs.append(mfcc)

		feature = np.mean(mfccs, axis=0) 

		y_true.append(vowel)

		predict_vowel = '#'
		min_dist = 1e20
		for v in vowels:
			for i in vowels_feature[v]:
				dist = np.linalg.norm(feature - i)
				if dist < min_dist:
					min_dist = dist
					predict_vowel = v

		y_pred.append(predict_vowel)

		if vowel == predict_vowel:
			ok += 1
		print(folder, vowel, predict_vowel)

print(ok / len(y_pred) * 100, '%')
plot_confusion_matrix(y_true, y_pred, vowels)
plt.show()








