categories = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

import os 
current_file = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.join(current_file, "data")

data = []
labels = []
sampling_rates = []

from scipy.io import wavfile

# Note: I removed jazz.00054.wav since it was causing an error
for category in categories:
    data_path = os.path.join(data_folder, "genres_original/" + category)
    files = os.listdir(data_path)
    for f in files:
        labels.append(category)
        print(f)
        sampling_rate, wav_data = wavfile.read(os.path.join(data_path, f))
        data.append(wav_data)
        sampling_rates.append(sampling_rate)

import matplotlib.pyplot as plt
#plt.plot(data[0])
#plt.show()

# TODO: Fourier Transform