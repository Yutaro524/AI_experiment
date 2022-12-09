import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re

paths = glob.glob(os.path.join("audio/*.wav"))
count = 0

for path in paths:
    plt.figure()
    c_dir, file_id, extension = re.split("[/.]", path)
    file_id_list = file_id.split("-")
    dir_name = "image/" + file_id_list[3]

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    img_filename = dir_name + "/" + file_id + ".png"

    y, sr = librosa.load(path) # y.shape:(117601,) sr:22050 

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, win_length=512, hop_length=512)

    ## dB単位に変換
    S_dB = librosa.power_to_db(S, ref=np.max)
    ## プロット
    img = librosa.display.specshow(S_dB, x_axis='time',
                            y_axis='mel', sr=sr)
    plt.colorbar(img, format='%+2.0f dB')

    plt.savefig(img_filename)

    plt.clf()
    plt.close()
    print(str(count) + "images are generated")
    count += 1

