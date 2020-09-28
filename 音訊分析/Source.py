import matplotlib.pyplot as plt
import numpy as np
import math
import librosa
import librosa.display


y, sr = librosa.load('sound.m4a')


frame_len = int(0.02 * sr)
frame_shift = int(0.01 * sr)
frameNum = math.ceil(len(y)/256)


###############################################################
plt.figure(figsize=(30, 5))
librosa.display.waveplot(y, sr=sr)
plt.title('Waveform')
plt.show()



###############################################################
zcr = np.zeros((frameNum,1))
for i in range(frameNum):
        curFrame = y[np.arange(i*256,min(i*256+256,len(y)))]
        zcr[i] = sum(curFrame[0:-1]*curFrame[1::]<=0)

zcr_time = np.arange(0, len(zcr)) * (len(y)/len(zcr) / sr)
plt.figure(figsize=(30, 5))
plt.plot(zcr_time, zcr)
plt.ylabel("Zero Crossing Rate")
plt.xlabel("Time (s)")
plt.title("Zero-crossing rate contour")
plt.show()



###############################################################
ste = np.zeros((frameNum,1))
for i in range(frameNum):
    curFrame = y[np.arange(i*256,min(i*256+256,len(y)))]
    ste[i] = sum(curFrame[0:-1]*curFrame[1::])

ste_time = np.arange(0, len(ste)) * (len(y)/len(ste) / sr)
plt.figure(figsize=(30, 5))
plt.plot(ste_time,ste)
plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.title("Energy Contour")
plt.grid('on')



###############################################################
frames = librosa.util.frame(y, frame_length=frame_len, hop_length=frame_shift)
pitches, magnitudes = librosa.core.piptrack(y, sr=sr, hop_length=frame_shift, threshold=0.75)

pitch_track = []
for i in range(0,pitches.shape[1]):
    pitch_track.append(np.max(pitches[:,i]))
    
x=np.r_[2*pitch_track[0]-pitch_track[10::-1],pitch_track,2*pitch_track[-1]-pitch_track[-1:-11:-1]]
temp = np.hanning(11)
temp = np.convolve(temp/temp.sum(),x,mode='same')
pitch_smoothtrack = temp[11:-10]
plt.figure(figsize=(30, 5))
plt.plot(pitch_smoothtrack)
plt.title("Pitch Contour")
plt.show()



###############################################################
root_mean_square_energy = librosa.feature.rms(y, frame_length=frame_len, hop_length=frame_shift)
rms = root_mean_square_energy[0]
rms = librosa.util.normalize(rms, axis=0)

zero_crossing_rate = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=frame_shift, threshold=0)
zcr = zero_crossing_rate[0]

frame_indexs = np.where( (rms > 0.3) | (zcr > 0.5) )[0]


start_indexs = [frame_indexs[0]]
end_indexs = []
index_shape = np.shape(frame_indexs)

for i in range(index_shape[0]-1):
    if (frame_indexs[i + 1] - frame_indexs[i]) != 1:
        start_indexs.append(frame_indexs[i+1])
        end_indexs.append(frame_indexs[i])

end_indexs.append(frame_indexs[-1])


start_indexs = np.array(start_indexs)
end_indexs = np.array(end_indexs)

start_temp = start_indexs * frame_shift / sr
end_temp = end_indexs * frame_shift / sr


plt.figure(figsize=(30, 5))
temp = np.linspace(0, len(y)/sr, len(y))
plt.plot(temp, y)
for start, end in zip(start_temp, end_temp):
    plt.axvline(x=start, color='#2ca02c') 
    plt.axvline(x=end, color='#d62728')
plt.title("End point detection")
plt.show()