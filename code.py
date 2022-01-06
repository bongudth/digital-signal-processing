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
	def Framing(self, frame_length = 0.025):
		frame_length = int(frame_length * self.Fs)
		frame_shift = frame_length // 4
		frame_num = int(len(self.x)/frame_shift + 1)
		
		return [frame_num, frame_length, frame_shift]

	# Get frames
	def GetFrames(self):
		frame_num, frame_length, frame_shift = self.Framing()
		frame_num = (len(self.x) - frame_length) // frame_shift
		
		frames = np.zeros(shape=(frame_num, frame_length))

		for i in range(frame_num):
			frames[i] = self.x[i * frame_shift : i * frame_shift + frame_length]

		return frames

	# Calculate the energy of each frame
	def Energy(self, frame):
		return np.sum(frame**2)

	# Calculate AMDF
	def CalcAMDF(self, x):
		N = len(x)
		x = np.concatenate((x, np.zeros(N)))

		AMDF = np.zeros(N)
		for i in range(N):
			AMDF[i] = np.sum(abs(x[0:N] - x[i:N+i]))
		
		# Normalize AMDF
		AMDF = AMDF / np.max(AMDF)

		return AMDF

	# Find fundamental frequency of a signal
	# Using ACF and AMDF
	def FundamentalFrequency_ACF_AMDF(self, T, t):
		frames = self.GetFrames()

		t_frames = []
		energy_max = 0

		for i in range(len(frames)):
			t_frames.append(t[i])
			energy_max = max(energy_max, self.Energy(frames[i]))

		t_frames = np.array(t_frames)

		F0_AMDF = np.zeros(len(frames))

		for i in range(len(frames)):
			lag_min = int(self.Fs / 400)
			lag_max = int(self.Fs / 70)

			if self.Energy(frames[i]) > T * energy_max:
				AMDF = self.CalcAMDF(frames[i])

				AMDF_lag = np.argmin(AMDF[lag_min : lag_max + 1]) + lag_min

				if AMDF[AMDF_lag] <= T:
					F0_AMDF[i] = self.Fs / AMDF_lag
				else:
					F0_AMDF[i] = 0

		return [t_frames, F0_AMDF]

	# Median filter to smooth the F0
	def MedianFilter(self, F0):
		F0_median = medfilt(F0, kernel_size = 5)

		# Calc F0 mean and std
		F0_filter = []
		for i in range(len(F0_median)):
			if F0_median[i] != 0:
				F0_filter.append(F0_median[i])
		F0_mean = np.mean(F0_filter)
		return [F0_median, F0_mean]
		
	def PlotFundamentalFrequency(self, nameFile):
		T = self.STEThreshold()
		t, _ = self.STE()

		t_frames, F0_AMDF = self.FundamentalFrequency_ACF_AMDF(T, t)
		F0_AMDF_median, F0_AMDF_mean = self.MedianFilter(F0_AMDF)
		
		fig = plt.figure(nameFile)
		plt.suptitle(nameFile)
		ax1 = fig.add_subplot(411)
		ax2 = fig.add_subplot(412)
		ax3 = fig.add_subplot(413)
		ax4 = fig.add_subplot(414)

		print('F0_AMDF: ', F0_AMDF_mean)
		if(F0_AMDF_mean > 150):
			print('Gender: female', end = '\n\n')
		else:
			print('Gender: male', end = '\n\n')

		# Plot F0_AMDF
		ax4.plot(t_frames, F0_AMDF_median, '.')
		ax4.set_title('AMDF')
		ax4.set_xlabel('Time (s)')
		ax4.set_ylabel('Frequency (Hz)')

		plt.tight_layout()
		# plt.savefig(nameFile[:-3] + 'png')

def main():
	name = ['04MHB.wav', '05MVB.wav', '07FTC.wav', '08MLD.wav', '09MPD.wav', '10MSD.wav', '12FTD.wav', '14FHH.wav', '16FTH.wav', '24FTL.wav']
	for i in name:
		wave = Wave(i)
		print("FILE" , i, end="\n\n")
		wave.PlotFundamentalFrequency(i)
		print("________________________________________", end="\n\n")
	plt.show()

if __name__ == '__main__':
	main()