import shutil

import cv2
import numpy as np
import os

import sys

if len(sys.argv) < 3:
    print("input params\npython myrename.py dir newdir")
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
print("index:" + str(nlen))

cv2.namedWindow("scene_change", cv2.WINDOW_NORMAL)
cv2.resizeWindow("scene_change", 800, 600)
for i in range(0, nlen):
    file_name, file_extend = os.path.splitext(dirs[i])
    if file_extend == '.mp4':
        video_path = os.path.join(source_dir, file_name + ".mp4")

        cap = cv2.VideoCapture(video_path)
        count = 0
        write_count = 0
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            last_frame_his_r = None
            last_frame_his_g = None
            last_frame_his_b = None
            video_write = None
            root_path = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # img = cv2.resize(frame, (400, 225))
                img = frame.copy()

                his_r = cv2.calcHist([img], [0], None, [256], [0, 256])
                his_r.shape = (256,)
                his_r_mod = np.linalg.norm(his_r)
                his_r = his_r / his_r_mod

                his_g = cv2.calcHist([img], [1], None, [256], [0, 256])
                his_g.shape = (256,)
                his_g_mod = np.linalg.norm(his_g)
                his_g = his_g / his_g_mod

                his_b = cv2.calcHist([img], [2], None, [256], [0, 256])
                his_b.shape = (256,)
                his_b_mod = np.linalg.norm(his_b)
                his_b = his_b / his_b_mod

                dis_r = 0.0
                dis_g = 0.0
                dis_b = 0.0
                if last_frame_his_r is not None:
                    dis_r = np.dot(last_frame_his_r, his_r)
                if last_frame_his_g is not None:
                    dis_g = np.dot(last_frame_his_g, his_g)
                if last_frame_his_b is not None:
                    dis_b = np.dot(last_frame_his_b, his_b)
                last_frame_his_r = his_r
                last_frame_his_g = his_g
                last_frame_his_b = his_b

                cv2.imshow("scene_change", img)
                cv2.waitKey(1)
                # print(dis_r, dis_g, dis_b)
                if dis_r < 0.95 or dis_g < 0.95 or dis_b < 0.95:
                    print("------------------", dis_r, dis_g, dis_b)
                    if video_write is not None:
                        video_write.release()
                        if root_path is not None and write_count < 60:
                            os.remove(root_path)
                    # count = count + 1
                    write_count = 0
                    create_dir = os.path.join(new_dir, file_name)
                    if not os.path.exists(create_dir):
                        os.makedirs(create_dir)
                    root_path = os.path.join(new_dir, file_name, file_name + "_" + "%06d" % count + ".mp4")
                    video_write = \
                        cv2.VideoWriter(root_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                if video_write is not None:
                    video_write.write(frame)
                    write_count = write_count + 1
                count = count + 1
