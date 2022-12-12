import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re

paths = glob.glob("OGVC/Acted/wav/*/*/*.wav")
count = 0
ax_w_px = 64  # プロット領域の幅をピクセル単位で指定
ax_h_px = 64  # プロット領域の高さをピクセル単位で指定

# サイズ指定のための処理 ↓↓ ここから ↓↓ 
fig_dpi = 100
ax_w_inch = ax_w_px / fig_dpi
ax_h_inch = ax_h_px / fig_dpi

fig = plt.figure( dpi=fig_dpi, figsize=(ax_w_inch, ax_h_inch))
# サイズ指定のための処理 ↑↑ ここまで ↑↑

for path in paths:

    _,_,_,person,AB,filename = path.split("/")
    pattern = "(\D+)(\d+)(\D+)(\d)(\D+)"
    result = re.match(pattern, filename)
    if result.group(4)=="0":
        continue
    dir_name = os.path.join("data/train/",person,AB,result.group(3))
    file_id = filename.strip(result.group(5))

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

    plt.cla()
    print(filename, str(count) + "images are generated")
    count += 1




