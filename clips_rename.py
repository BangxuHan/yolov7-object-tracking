# coding:utf-8
import os
import shutil
import sys

if len(sys.argv) < 4:
    print("input params\npython myrename.py dir newdir startnumber")
    sys.exit()
source_dir = sys.argv[1]
new_dir = sys.argv[2]
# Do not use same value as path
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

dirs = os.listdir(source_dir)
dirs.sort()
print(type(dirs))
nlen = len(dirs)
index = int(sys.argv[3]) + 1
print("index:" + str(nlen))
txt_path = os.path.join(source_dir, "name_map.txt")
with open(txt_path, "a+") as txt_file:
    for i in range(0, nlen):
        file_name, file_extend = os.path.splitext(dirs[i])
        if file_extend == '.mp4' or file_extend == '.MP4':
            oldnamemp4 = os.path.join(source_dir, file_name + ".mp4")
            newnamemp4 = os.path.join(new_dir, "%06d" % index + ".mp4")
            shutil.copyfile(oldnamemp4, newnamemp4)
            txt_file.write(file_name + ".mp4" + "--->" + "%06d" % index + ".mp4"+"\n")
            index = index + 1
            print(oldnamemp4 + "--->" + newnamemp4)
