from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal.signaltools import medfilt

class Wave:

	# Construtor
	def __init__(self, nameFile):
		self.Fs, self.x = wavfile.read(nameFile)

	@property 
	def times(self):
		return np.arange(0, len(self.x)/self.Fs, 1/self.Fs)

	# Normalize x
	def Normalize(self):
		self.x = np.array(self.x)/max(abs(self.x))

	# Calculate Short-Time Energy
	def STE(self, N = 0.025):
		self.Normalize()

		N = N * self.Fs
		step = int(N // 4)

		E = np.zeros(len(self.x) + 1)

		for i in range(1, len(self.x) + 1):
			E[i] = E[i - 1] + self.x[i - 1]**2

		ste = []
		t = []

		for i in range(1, len(E), step):
			start = int(i - N // 2 + 1)
			end = int(i + N // 2)
			ste.append(E[min(len(E) - 1, end)] - E[max(1, start) - 1])
			t.append((i - 1) / self.Fs)

		# Normalize STE
		ste = np.array(ste)/max(ste)
		t = np.array(t)

		return [t, ste]

	#	Init threshold value
	def InitThreshold(self, height = 1e-2):
		t, ste = self.STE()
		pre = 0
		i = 1
		tmp = []

		while i < len(ste):
			while i < len(ste) - 1 and ste[i] >= ste[i - 1]:
				i = i + 1
			top = i - 1

			while i < len(ste) - 1 and ste[i] < ste[i - 1]:
				i = i + 1
			cur = i - 1

			if ste[top] - ste[pre] >= height or ste[top] - ste[cur] >= height:
				tmp.append(ste[top])
			pre = cur
			i = i + 1

		tmp.sort()
		return tmp[0]

	# Array of index speech and silence
	def DetectSilenceSpeech(self, T, minSilenceLen = 0.3):
		t, ste = self.STE()
		silence = []
		speech = []
		tmpSil = []

		for i in range(len(ste)):
			if ste[i] < T:
				tmpSil.append(i)
			else:
				if len(tmpSil) > 0 and t[tmpSil[-1]] - t[tmpSil[0]] >= minSilenceLen:
					silence.append(np.array(tmpSil))
				tmpSil = []

		if len(tmpSil) > 0:
			silence.append(np.array(tmpSil))

		start = 0
		for segment in silence:
			end = segment[0] - 1

			for i in range(start, end + 1):
				speech.append(i)
			start = segment[-1] + 1

		for i in range(start, len(ste)):
			speech.append(i)
		speech = np.array(speech)

		return [silence, speech]

	# Array of Overlap Speech and Overlap Silence
	def DetectOverlap(self, T):
		t, ste = self.STE()
		silence, speech = self.DetectSilenceSpeech(T)

		Tmin = 1
		Tmax = 0

		for segment in silence:
			for i in segment:
				Tmax = max(Tmax, ste[i])

		for i in speech:
			Tmin = min(Tmin, ste[i])

		f = []
		g = []

		for segment in silence:
			for i in segment:
				if ste[i] >= Tmin and ste[i] <= Tmax:
					f.append(ste[i])

		for i in speech:
			if ste[i] >= Tmin and ste[i] <= Tmax:
				g.append(ste[i])

		f = np.array(f)
		g = np.array(g)

		return [f, g]

	# 1. From the overlapping attribute functions, only keep the overlapping part, that you store in vectors f and g

	# Calculate STE threshold use Binary Search
	def STEThreshold(self):
		T = self.InitThreshold()
		f, g = self.DetectOverlap(T)

		if len(f) == 0 or len(g) == 0:
			return T

		# 2. Set Tmin and Tmax as the minimum and maximum of this region of overlap
		Tmin = min(min(g), min(f))
		Tmax = max(max(g), max(f))

		# 3. Set T = 1/2(Tmin + Tmax)
		Tmid = (Tmax + Tmin)/2

		# 4. Set i and p as the number of values of f and g below and above T
		i = len([i for i in f if i < Tmid])
		p = len([i for i in g if i > Tmid])

		# 5. Set j and q both to -1
		j = -1
		q = -1

		# 6. So long as i != j or p != q, repeat the following steps
		while i != j or p != q:
			value = sum([max(i - Tmid, 0) for i in f])/len(f) - sum([max(Tmid - i, 0) for i in g])/len(g)

			# 7. If it is positive, set Tmin = T. Else, set Tmax = T
			if value > 0:
				Tmin = Tmid
			else:
				Tmax = Tmid

			# 8. Set T = 1/2 (Tmin + Tmax)
			Tmid = (Tmax + Tmin)/2
			
			# 9. Set j = i, and q = p
			j = i
			q = p

			# 10. Set i = f < T and p = g > T
			i = len([i for i in f if i < Tmid])
			p = len([i for i in g if i > Tmid])

		return Tmid

	# Framing signal
	def Framing(self, frame_length):
		frame_length = int(frame_length * self.Fs)
		frame_shift = frame_length // 4
		frame_num = int(len(self.x)/frame_shift + 1)
		
		return [frame_num, frame_length, frame_shift]

	# Calculate Hamming
	def Hamming(self, N):
		return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))

	# Find FFT points
	def FindFFTPoints(self):
		N_FFT = 2**(int(np.log2(self.Fs/2)) + 1)
		print("FFT points: ", N_FFT)
		return N_FFT

	# Finding fundamental frequency of a signal
	def FundamentalFrequency(self, speech, frame_length = 0.025):
		self.Normalize()
		N_FFT = self.FindFFTPoints()		
		frame_num, frame_length, frame_shift = self.Framing(frame_length)
		h = self.Hamming(frame_length)

		peak_index = []
		peaks = []
		F0_FFT = np.zeros(frame_num)
		
		# The frequency of each spectrum
		freq = np.fft.fftfreq(N_FFT, 1/self.Fs)
		one_sided_freq = freq[:N_FFT//2]
			
		# Loop for each speech frame
		for (pos, i) in enumerate(speech):

			# Index of each element in window
			index = np.arange(i * frame_shift, i * frame_shift + frame_length)

			# Value of each element in window
			value = self.x[index] * h

			# Use FFT function to analyze the spectrum of the frame
			# The two-sided spectrum
			two_sided_spectrum = abs(np.fft.fft(value, N_FFT))

			# The one-sided spectrum
			one_sided_spectrum = two_sided_spectrum[0:N_FFT//2]
			one_sided_spectrum[1:] = one_sided_spectrum[1:] * 2

			# The index of peaks
			peak_index = find_peaks(one_sided_spectrum, height=4, prominence=4, distance=70)[0]
			if len(peak_index) <= 3:
				continue
			
			peakFirst = one_sided_freq[peak_index[0]]
			peakSecond = one_sided_freq[peak_index[1]]
			peakThird = one_sided_freq[peak_index[2]]

			f0_temp = abs(peakFirst - peakSecond)
			f1_temp = abs(peakSecond - peakThird)

			if f0_temp > 70 and f0_temp < 400:
				if f1_temp > 70 and f1_temp < 400:
					F0_FFT[i] = (f0_temp + f1_temp)/2

			if (pos == 6):
				# Get 3 peak in peaks
				peaks.append(peak_index[:3])

		return F0_FFT

	# Median filter to smooth the F0
	def MedianFilter(self, F0):
		F0_median = medfilt(F0, kernel_size = 5)

		# Calc F0 mean and std
		F0_filter = []
		for i in range(len(F0_median)):
			if F0_median[i] != 0:
				F0_filter.append(F0_median[i])
		F0_mean = np.mean(F0_filter)
		F0_std = np.std(F0_filter)
		return [F0_median, F0_mean, F0_std]
		
	def PlotFundamentalFrequency(self, nameFile):
		T = self.STEThreshold()
		print('Threshold: ', T, end = '\n\n')

		_, g = self.DetectSilenceSpeech(T)
		t, _ = self.STE()
		F0_FFT = self.FundamentalFrequency(g)
		F0_FFT_median, F0_FFT_mean, F0_FFT_std = self.MedianFilter(F0_FFT)
		
		fig = plt.figure(nameFile)
		plt.suptitle(nameFile)
		ax1 = fig.add_subplot(411)
		ax2 = fig.add_subplot(412)
		ax3 = fig.add_subplot(413)
		ax4 = fig.add_subplot(414)

		print(">> Student")

		print('F0mean: ', F0_FFT_mean)
		print('F0std: ', F0_FFT_std, end = '\n\n')

		file = open(nameFile[:-3] + "txt", "r")
		
		print(">> Teacher")

		for i in file:
			i = i.split()

			if i[0] == 'F0mean':
				print("F0mean: ", i[1])

			if i[0] == 'F0std':
				print("F0std: ", i[1])

		# Plot F0 and silence speech discrimination
		ax1.plot(t, F0_FFT_median, '.')
		ax1.set_title('FFT')
		ax1.set_xlabel('Time (s)')
		ax1.set_ylabel('Frequency (Hz)')

		plt.tight_layout()
		# plt.savefig(nameFile[:-3] + 'png')

def main():
	name = ['01MDA.wav', '02FVA.wav', '03MAB.wav', '06FTB.wav', '30FTN.wav', '42FQT.wav', '44MTT.wav', '45MDV.wav']
	for i in name:
		wave = Wave(i)
		print("FILE" , i, end="\n\n")
		wave.PlotFundamentalFrequency(i)
		print("________________________________________", end="\n\n")
	plt.show()

if __name__ == '__main__':
	main()