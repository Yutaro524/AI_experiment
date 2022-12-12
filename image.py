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

    ax_w_px = 64  # プロット領域の幅をピクセル単位で指定
    ax_h_px = 64  # プロット領域の高さをピクセル単位で指定

    # サイズ指定のための処理 ↓↓ ここから ↓↓ 
    fig_dpi = 100
    ax_w_inch = ax_w_px / fig_dpi
    ax_h_inch = ax_h_px / fig_dpi

    fig = plt.figure( dpi=fig_dpi, figsize=(ax_w_inch, ax_h_inch))
    # サイズ指定のための処理 ↑↑ ここまで ↑↑

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
    img = librosa.display.specshow(S_dB, sr=sr)


    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.savefig(img_filename)

    plt.clf()
    plt.close()
    print(str(count) + "images are generated")
    count += 1

