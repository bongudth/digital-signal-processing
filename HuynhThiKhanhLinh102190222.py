from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from math import log

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

		#Normalize STE
		ste = np.array(ste)/max(ste)
		t = np.array(t)

		return [t, ste]

	#	Init threshold value
	def InitThreshold(self, height = 1e-3):
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

	def PlotSpeechSilentDiscrimination(self, nameFile):
		n = self.times
		T = self.STEThreshold()
		f, g = self.DetectSilenceSpeech(T)
		t, ste = self.STE()

		fig = plt.figure(nameFile)
		plt.suptitle(nameFile)
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)

		print(">> Student")

		for i in f:
			start, end = t[i[0]], t[i[-1]]

			ax1.plot([start, start], [0, max(ste)], '#008000')
			ax1.plot([end, end], [0, max(ste)], '#008000')
			ax1.set_title('Short-Time Energy')
			ax1.set_xlabel('Time (s)')
			ax1.set_ylabel('Enegry')

			ax2.plot([start, start], [-1, 1], '#008000')
			ax2.plot([end, end], [-1, 1], '#008000')
			ax2.set_title('Speech Silent Discrimination')
			ax2.set_xlabel('Time (s)')
			ax2.set_ylabel('Amplitude')

			print(start, "\t", end)

		print()

		file = open(nameFile[:-3] + "txt", "r")
		
		print(">> Teacher")

		for i in file:
			i = i.split()
			if i[-1] == 'sil':
				start, end = float(i[0]), float(i[1])

				print(start, "\t\t\t\t", end)

				ax1.plot([start, start], [0, max(ste)], '#FF0000')
				ax1.plot([end, end], [0, max(ste)], '#FF0000')

				ax2.plot([start, start], [-1, 1], '#FF0000')
				ax2.plot([end, end], [-1, 1], '#FF0000')

		ax1.plot([0, n[-1]], [T, T], '#FFA500')
		ax1.plot(t, ste, '#0080FF')

		data = self.x

		ax2.plot(n, data, '#0080FF')
		ax2.plot(t, ste, '#FF0000')

		plt.tight_layout()
		plt.savefig(nameFile[:-3] + 'png')

def main():
	name = ['30FTN.wav', '42FQT.wav', '44MTT.wav', '45MDV.wav']
	for i in name:
		wave = Wave(i)
		print("FILE" , i, end="\n\n")
		wave.PlotSpeechSilentDiscrimination(i)
		print("________________________________________", end="\n\n")
	plt.show()

if __name__ == '__main__':
	main()