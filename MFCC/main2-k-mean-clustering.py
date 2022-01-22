def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

'''
	1.	PHÂN ĐOẠN TÍN HIỆU THÀNH NGUYÊN ÂM / KHOẢNG LẶNG
'''

import numpy as np
import soundfile as sf

# các file tín hiệu huấn luyện
training_file_names = ['01MDA', '02FVA', '03MAB', '06FTB', '30FTN', '42FQT', '44MTT', '45MDV']

# độ dài khung tín hiệu (20ms)
frame_time = 0.02

# độ dịch khung tín hiệu (10ms)
frame_shift_time = 0.01

def read_file_lab(file_name):
	voice = []
	unvoice = []
	file = open(file_name, 'r')
	for line in file:
		data = line.split()
		if len(data) == 3:
			if data[-1] != 'sil':
				voice.append([float(data[0]), float(data[1])])
			else:
				unvoice.append([float(data[0]), float(data[1])])
	return voice, unvoice

def MA(signal, fs):
	signal_length = len(signal)
	frame_length = int(frame_time * fs)
	frame_shift = int(frame_shift_time * fs)
	ma = []
	for i in range(1, int(signal_length / frame_shift) - int(frame_length / frame_shift) + 3):
		frame = signal[(i-1)*frame_shift+1 : frame_shift*i + frame_length]
		ma.append(sum(abs(frame)))
	return ma / max(ma)

def calculate_threshold(file_names):
	
	threshold = np.array([])

	for file_name in file_names:
		signal, fs = sf.read(file_name + '.wav', dtype=np.int16)
		voice, unvoice = read_file_lab(file_name + '.txt')
		ma = np.array(MA(signal, fs))

		mean_voice = []
		mean_unvoice = []
		std_voice = []
		std_unvoice = []

		for i in range(0, len(voice)):
			arr = np.arange(int(voice[i][0] * 100), int(voice[i][1] * 100), step=1)
			mean_voice = np.append(mean_voice,np.mean(ma[arr]))
			std_voice = np.append(std_voice,np.std(ma[arr]))
		for i in range(0, len(unvoice)):
			arr = np.arange(int(unvoice[i][0] * 100), int(unvoice[i][1] * 100-3), step=1)
			mean_unvoice = np.append(mean_unvoice,np.mean(ma[arr]))
			std_unvoice = np.append(std_unvoice,np.std(ma[arr]))

		sum = 0
		for i in range(0, len(mean_voice)):
			sum = mean_voice[i] + sum

		meanV = np.average(mean_voice)
		meanUV = np.average(mean_unvoice)
		stdV = np.average(std_voice)
		stdUV = np.average(std_unvoice)

		threshold = np.append(threshold, (meanV - stdV + meanUV + stdUV) / 2)

	return np.average(threshold)

threshold = calculate_threshold(training_file_names)

'''
	2.	TRÍCH XUẤT VECTOR ĐẶC TRƯNG ĐƯỜNG BAO PHỔ (SPECTRAL ENVELOPE) CỦA 5 NGUYÊN ÂM DỰA TRÊN TẬP HUẤN LUYỆN
'''

import os
import matplotlib.pyplot as plt
import librosa
from sklearn.cluster import KMeans

# đường dẫn đến folder nguyên âm huấn luyện
training_folder = './NguyenAmHuanLuyen-16k'

# lấy danh sách tên các folder trong nguyên âm huấn luyện
training_vowel_folders = []
for i in os.listdir(training_folder):
	training_vowel_folders.append(i)

# các nguyên âm
vowel_files = ['a', 'e', 'i', 'o', 'u']

# số chiều của một khung tín hiệu (số lượng hệ số MFCC)
N = 13 # 13, 26, 39

# số vector đặc trưng cho 1 nguyên âm của nhiều người nói (K-means clustering)
K = 9 # 2, 3, 4, 5

def get_frames(signal, fs, t):
	frame_length = int(frame_time * fs)
	frame_shift = int(frame_shift_time * fs)

	frames = []
	tframes = []

	for i in range(frame_length // 2, len(signal) - frame_length // 2, frame_shift):
		frames.append(np.array(signal[i - frame_length // 2: i + frame_length // 2]))
		tframes.append(t[i])

	frames = np.array(frames)
	tframes = np.array(tframes)
	return [frames, tframes]

def get_speech(ma, tma):
	t = tma[ma >= threshold]
	line = [t[0]]
	for i in range(1, len(t)):
		if t[i] - t[i - 1] > 0.01:
			line = np.append(line, [t[i - 1], t[i]])

	speech = []
	tmp = []
	for i in range(len(ma)):
		if ma[i] > threshold:
			tmp.append(i)
		else:
			if len(tmp) > 0 and tma[tmp[-1]] - tma[tmp[0]] > 0.17:
				speech.append(tmp)
			tmp = []

	res = []
	for segment in speech:
		res.append([segment[0], segment[-1]])

	return res

vowel_mfccs_array = {}
wovel_kmeans_clustering_array = {}
for i in vowel_files:
	vowel_mfccs_array[i] = []
	wovel_kmeans_clustering_array[i] = []

for file in vowel_files:
	mfcc_vector = []
	for folder in training_vowel_folders:
		signal, fs = sf.read(training_folder + '/' + folder + '/' + file + '.wav', dtype=np.int16)
		signal_length = len(signal)
		frame_length = int(frame_time * fs)
		frame_shift = int(frame_shift_time * fs)

		signal = np.array(signal)
		t = np.linspace(0.0, signal_length / fs, signal_length)
		signal = signal / max(abs(signal))
		frames, tframes = get_frames(signal, fs, t)

		ma = MA(signal, fs)

		tFrame = (
				np.linspace(	
					frame_shift/2,
					signal_length - frame_shift/2,
					int(signal_length/frame_shift) - int(frame_length/frame_shift) + 2
				)
			)/fs

		fig = plt.figure()
		sub1 = fig.add_subplot(111)

		# vẽ tín hiệu ban đầu

		sub1.plot(t, signal)
		sub1.set_title(folder + ' - ' + file)
		sub1.set_xlabel('Time (s)')
		sub1.set_ylabel('Amplitude')

		# kẻ các biên voice tìm được

		speech = get_speech(ma, tFrame)
		for i in speech:
			start, end = tFrame[i[0]], tFrame[i[-1]]
			sub1.plot([start, start], [-1, 1], 'red')
			sub1.plot([end, end], [-1, 1], 'red')

		'''
			a. Đánh dấu vùng có đặc trưng phổ ổn định đặc trưng cho nguyên âm:
				chia vùng nguyên âm thành 3 phần bằng nhau và lấy đoạn nằm giữa (gồm M khung)
		'''
		
		start, end = speech[0]
		vowel_length = end - start
		third = vowel_length // 3
		left, right = start + third, end - third
		l, r = tFrame[left], tFrame[right]
		sub1.plot([l, l], [-1, 1], 'orange')
		sub1.plot([r, r], [-1, 1], 'orange')

		'''
			b. Trích xuất vector MFCC (mel-frequency cepstral coefficients) của 1 khung tín hiệu với số chiều (N)
		'''

		for i in range(left, right + 1):
			mfccs = librosa.feature.mfcc(y=frames[i], sr=fs, n_mfcc=N)
			tmp = []
			for j in mfccs:
				tmp.append(*j)
			mfcc_vector.append(np.array(tmp))

	vowel_mfccs_array[file].append(mfcc_vector)

for v in vowel_files:
	kmeans = KMeans(n_clusters=K, random_state=0).fit(vowel_mfccs_array[v][0]).cluster_centers_
	wovel_kmeans_clustering_array[v] = kmeans

print('wovel_kmeans_clustering_array:', wovel_kmeans_clustering_array)

'''
	3.	SO KHỚP VECTOR MFCC CỦA TÍN HIỆU NGUYÊN ÂM ĐẦU VÀO VỚI 5*K VECTOR ĐẶC TRUNG ĐÃ TRÍCH XUẤT CỦA 5 NGUYÊN ÂM
		ĐƯA RA KẾT QUẢ NHẬN DẠNG NGUYÊN ÂM: TÍNH 5*K KHOẢNG CÁCH EUCLIDEAN GIỮA 2 VECTOR VÀ ĐƯA RA QUYẾT ĐỊNH
		NHẬN DẠNG TRÊN KHOẢNG CÁCH NHỎ NHẤT
'''

# đường dẫn đến folder nguyên âm kiểm thử
testing_folder = './NguyenAmKiemThu-16k'

# lấy danh sách tên các folder trong nguyên âm kiểm thử
testing_vowel_folders = []
for i in os.listdir(testing_folder):
	testing_vowel_folders.append(i)

# khởi tạo ma trận nhầm lẫn
confusion_matrix = {}
for v in vowel_files:
	confusion_matrix[v] = {}
	for w in vowel_files:
		confusion_matrix[v][w] = 0

total_files = 0
error_files = 0
for file in vowel_files:
	for folder in testing_vowel_folders:
		total_files += 1
		signal, fs = sf.read(testing_folder + '/' + folder + '/' + file + '.wav', dtype=np.int16)
		signal_length = len(signal)
		frame_length = int(frame_time * fs)
		frame_shift = int(frame_shift_time * fs)

		signal = np.array(signal)
		t = np.linspace(0.0, signal_length / fs, signal_length)
		signal = signal / max(abs(signal))
		frames, tframes = get_frames(signal, fs, t)

		ma = MA(signal, fs)

		tFrame = (
				np.linspace(	
					frame_shift/2,
					signal_length - frame_shift/2,
					int(signal_length/frame_shift) - int(frame_length/frame_shift) + 2
				)
			)/fs

		speech = get_speech(ma, tFrame)

		# lấy ra đoạn phần ba ở giữa khoảng nguyên âm
		
		start, end = speech[0]
		vowel_length = end - start
		third = vowel_length // 3
		left, right = start + third,  end - third

		mfcc_vector = []
		for i in range(left, right + 1):
			mfccs = librosa.feature.mfcc(y=frames[i], sr=fs, n_mfcc=N)
			tmp = []
			for j in mfccs:
				tmp.append(*j)
			mfcc_vector.append(np.array(tmp))

		mfccs_average_array = sum(mfcc_vector) / len(mfcc_vector)

		euclidean = 1000
		presumable_results = ''
		for v in wovel_kmeans_clustering_array:
			for i in range(0, K):
				new_euclidean = np.linalg.norm(mfccs_average_array - wovel_kmeans_clustering_array[v][i])
				if new_euclidean < euclidean:
					euclidean = new_euclidean
					presumable_results = v
		
		# nếu nhận dạng sai
		if file != presumable_results:
			error_files += 1

		confusion_matrix[file][presumable_results] += 1

print('\n\nN =', N)
print('K =', K)
print('Total Files:', total_files)
print('Error Files:', error_files)
print('Accuracy:', (total_files - error_files) / total_files * 100, '%\n')

print('\n\n 					Ma trận nhầm lẫn\n\n')
print(f"{'-'*63: ^63}")
print(f"|{'': ^21}|{'Nhãn dự đoán': ^39}|")
print(f"|{'': ^21}|{'-'*39: ^39}|")
print(f"|{'': ^21}|{'/a/': ^7}|{'/e/': ^7}|{'/i/': ^7}|{'/o/': ^7}|{'/u/': ^7}|")
print(f"|{'-'*61: ^61}|")
print(f"|{'': ^13}|{'/a/': ^7}|{confusion_matrix['a']['a']: ^7}|{confusion_matrix['a']['e']: ^7}|{confusion_matrix['a']['i']: ^7}|{confusion_matrix['a']['o']: ^7}|{confusion_matrix['a']['u']: ^7}|")
print(f"|{'': ^13}|{'': ^7}|{'-'*39: ^39}|")
print(f"|{'': ^13}|{'/e/': ^7}|{confusion_matrix['e']['a']: ^7}|{confusion_matrix['e']['e']: ^7}|{confusion_matrix['e']['i']: ^7}|{confusion_matrix['e']['o']: ^7}|{confusion_matrix['e']['u']: ^7}|")
print(f"|{'': ^13}|{'': ^7}|{'-'*39: ^39}|")
print(f"|{'Nhãn đúng': ^13}|{'/i/': ^7}|{confusion_matrix['i']['a']: ^7}|{confusion_matrix['i']['e']: ^7}|{confusion_matrix['i']['i']: ^7}|{confusion_matrix['i']['o']: ^7}|{confusion_matrix['i']['u']: ^7}|")
print(f"|{'': ^13}|{'': ^7}|{'-'*39: ^39}|")
print(f"|{'': ^13}|{'/o/': ^7}|{confusion_matrix['o']['a']: ^7}|{confusion_matrix['o']['e']: ^7}|{confusion_matrix['o']['i']: ^7}|{confusion_matrix['o']['o']: ^7}|{confusion_matrix['o']['u']: ^7}|")
print(f"|{'': ^13}|{'': ^7}|{'-'*39: ^39}|")
print(f"|{'': ^13}|{'/u/': ^7}|{confusion_matrix['u']['a']: ^7}|{confusion_matrix['u']['e']: ^7}|{confusion_matrix['u']['i']: ^7}|{confusion_matrix['u']['o']: ^7}|{confusion_matrix['u']['u']: ^7}|")
print(f"{'-'*63: ^63}")

plt.show()
