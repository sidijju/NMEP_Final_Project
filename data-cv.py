import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import cv2

categories = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

import os 
current_file = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(current_file, "data")

data = []
labels = []
sampling_rates = []


# Note: I removed jazz.00054.wav since it was causing an error
for category in categories:
    data_path = os.path.join(data_folder, "genres_original/" + category)
    files = os.listdir(data_path)
    for f in files:
        labels.append(category)
        #print(f)
        sampling_rate, wav_data = wavfile.read(os.path.join(data_path, f))
        #print(sampling_rate, wav_data)
        frequencies, times, spectrogram = signal.spectrogram(wav_data, sampling_rate)
        
        ''' 
        #Visualizes spectrogram data
        plt.pcolormesh(times, frequencies, np.log(spectrogram))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        '''
        #This writes spectrogram data to files, but loses decimal data
        #cv2.imwrite('./data/wav_to_spectrogram/'+f[:-4]+'.jpg', np.log(spectrogram))
        
        data.append(np.log(spectrogram))
        sampling_rates.append(sampling_rate)


# TODO: Fourier Transform
