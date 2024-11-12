import os
import cv2


def read(save_path):
    save_files = os.listdir(save_path)
    save_files.sort()

    for i in save_files:
        video_path = os.path.join(save_path, i)
        id = i.split('_')[1]
        new_num = '000045_' + id
        print(new_num)
        os.rename(i, new_num)
        # cap = cv2.VideoCapture(video_path)
        # # 获取视频的总帧数
        # frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #
        # # 获取视频的fps
        # fps = int(cap.get(cv2.CAP_PROP_FPS))
        #
        # print(i, fps)


if __name__ == '__main__':
    save_path = "/mnt/marathon/pingpangclips/migu2022/video_rename_no_scene_change/000042"

    read(save_path)
