import os
import sys


root = '/mnt/marathon/pingpangclips/migu2022/video_rename_no_scene_change/000060'

filelist = os.listdir(root)
filelist.sort()
currentpath = os.getcwd()
os.chdir(root)

for i in filelist:
    id = i.split('_')[1]
    new_num = '000057_' + id
    print(i + ' ---> ' + new_num)
    os.rename(i, new_num)

os.chdir(currentpath)
sys.stdin.flush()
print('update successful')
