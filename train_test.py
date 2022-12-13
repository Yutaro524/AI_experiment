import os
import shutil
import glob

dir_train = "data/train"
dir_test = "data/test"
classes = {
    "ACC":0, "ANG":1, "ANT":2, "DIS":3, "FEA":4, "JOY":5, "SAD":6, "SUR":7
    #データセットを参照して適切な辞書を作成する
}
p_list = ["F1", "F2", "M1", "M2"]
AB_list = ["A", "B"]
for p in p_list:
    for AB in AB_list:
        for cls in classes.keys():
            img_list = os.listdir(os.path.join(dir_train, p, AB, cls))[-6:]
            for j in img_list:
                filename1 = os.path.join(dir_train, p, AB, cls, j)
                filename2 = os.path.join(dir_test, p, AB, cls)
                if not os.path.isdir(filename2):
                    os.makedirs(filename2)
                shutil.move(filename1, filename2)
