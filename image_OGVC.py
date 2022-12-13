import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re
from augmentation import calculate_melsp, add_white_noise, shift_sound, stretch_sound

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
    img_filename1 = dir_name + "/" + file_id + "white.png"
    img_filename2 = dir_name + "/" + file_id + "shift.png"
    img_filename3 = dir_name + "/" + file_id + "stretch.png"

    y, sr = librosa.load(path) # y.shape:(117601,) sr:22050 

    y1 = add_white_noise(y)

    y2 = shift_sound(y)

    y3 = stretch_sound(y)

    ## dB単位に変換
    S_dB = calculate_melsp(y)
    ## プロット
    img = librosa.display.specshow(S_dB, sr=sr)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.savefig(img_filename)

    plt.cla()

    ## dB単位に変換
    S_dB = calculate_melsp(y1)
    ## プロット
    img = librosa.display.specshow(S_dB, sr=sr)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.savefig(img_filename1)

    plt.cla()

    ## dB単位に変換
    S_dB = calculate_melsp(y2)
    ## プロット
    img = librosa.display.specshow(S_dB, sr=sr)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.savefig(img_filename2)

    plt.cla()

    ## dB単位に変換
    S_dB = calculate_melsp(y3)
    ## プロット
    img = librosa.display.specshow(S_dB, sr=sr)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    plt.savefig(img_filename3)

    plt.cla()


    print(filename, str(count) + "images are generated")
    count += 1




