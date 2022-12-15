
import glob
import os
import re
import shutil
paths = glob.glob("OGVC/Acted/wav/*/*/*.wav")
count = 0

for path in paths:

    _,_,_,person,AB,filename = path.split("/")
    pattern = "(\D+)(\d+)(\D+)(\d)(\D+)"
    result = re.match(pattern, filename)
    if result.group(4)=="0":
        continue
    dir_name = os.path.join("data/emotion-Corpus/wav48",result.group(3))
    file_id = filename.strip(result.group(5))

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    new_filename = dir_name + "/" + file_id + ".wav"

    shutil.copyfile(path, new_filename)