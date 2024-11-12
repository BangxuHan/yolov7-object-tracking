import cv2
import torch

from deep_sort import build_tracker
from models.experimental import attempt_load
# For SORT tracking
from sort import *
from utils.datasets import letterbox
from utils.download_weights import download
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, strip_optimizer, set_logging
from utils.torch_utils import select_device, time_synchronized, TracedModel

# ............................... Tracker Functions ............................
""" Random created palette"""
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

"""" Calculates the relative bounding box from absolute pixel values. """


def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


"""Function to Draw Bounding boxes"""


def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
        label = str(id) + ":" + names[cat]
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 20), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 144, 30), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, [255, 255, 255], 1)
        # cv2.circle(img, data, 6, color,-1)
    return img


# ..............................................................................
def process_single_video(source_path, save_dir, file_name, detect_model,
                         imgsz, device, half, names, view_img, min_number_frame):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    width_add = 0
    height_add = 0
    deep_sort = build_tracker(use_cuda=True, age=10)
    cap = cv2.VideoCapture(source_path)
    assert cap.isOpened(), f'Failed to open {source_path}'
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) % 100
    save_single_object = {}
    frame_number = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        # Padded resize
        img = letterbox(image, imgsz, stride=32)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = detect_model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s = f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # ..................USE TRACK FUNCTION....................
                np_det = det.cpu().detach().numpy()
                new_bbox = xyxy2xywh(np_det[:, 0:4].astype(np.float32))
                cls_conf = np_det[:, 4].astype(np.float32)
                clss = np_det[:, 5].astype(np.float32)

                # Run SORT
                # tracked_dets, delete_track_ids = sort_tracker.update(dets_to_sort)
                # Run DeeepSort
                tracked_dets, delete_track_ids = deep_sort.update(new_bbox, cls_conf, image)

                # save video for one object
                tracks = deep_sort.tracker.tracks

                # for track_det in tracked_dets:
                #     bbox_xyxy = track_det[:4]
                #     identitie = track_det[4]
                print("88888888888888888888888888888", save_single_object)
                for track in tracks:
                    bbox_xyxy = track.to_tlbr().astype(np.int32)
                    identitie = track.track_id
                    if identitie in save_single_object.keys():
                        tmp_img = np.zeros((h, w, 3), dtype=np.uint8)
                        x1 = bbox_xyxy[0] - width_add
                        if x1 < 0:
                            x1 = 0
                        y1 = bbox_xyxy[1] - height_add
                        if y1 < 0:
                            y1 = 0
                        x2 = bbox_xyxy[2] + width_add
                        if x2 > w - 1:
                            x2 = w - 1
                        y2 = bbox_xyxy[3] + height_add
                        if y2 > h - 1:
                            y2 = h - 1
                        tmp_img[y1:y2, x1:x2] = \
                            image[y1:y2, x1:x2]
                        save_single_object[identitie]["video_write"].write(tmp_img)
                        save_single_object[identitie]["write_count"] += 1
                        print(identitie, "---------------------------", save_single_object[identitie]["write_count"])
                    else:
                        file_name_all = file_name + "_" + "FN%06d" % frame_number + "_TID%06d" % identitie + ".avi"
                        save_path = os.path.join(save_dir, file_name_all)
                        save_single_object[identitie] = \
                            {
                                "video_write": cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h)),
                                "write_count": 0,
                                "save_path": save_path
                            }
                        print(save_single_object)

                        tmp_img = np.zeros((h, w, 3), dtype=np.uint8)
                        x1 = bbox_xyxy[0] - width_add
                        if x1 < 0:
                            x1 = 0
                        y1 = bbox_xyxy[1] - height_add
                        if y1 < 0:
                            y1 = 0
                        x2 = bbox_xyxy[2] + width_add
                        if x2 > w - 1:
                            x2 = w - 1
                        y2 = bbox_xyxy[3] + height_add
                        if y2 > h - 1:
                            y2 = h - 1
                        tmp_img[y1:y2, x1:x2] = \
                            image[y1:y2, x1:x2]
                        save_single_object[identitie]["video_write"].write(tmp_img)
                        save_single_object[identitie]["write_count"] += 1
                for del_id in delete_track_ids:
                    if del_id in save_single_object.keys():
                        save_single_object[del_id]["video_write"].release()
                        if save_single_object[del_id]["write_count"] < min_number_frame:
                            try:
                                print("del  ", save_single_object[del_id]["save_path"])
                                os.remove(save_single_object[del_id]["save_path"])
                            except:
                                # 报错不做任何处理
                                pass
                        save_single_object.pop(del_id)

                # draw boxes for visualization
                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:, :4]
                    identities = tracked_dets[:, 4]
                    draw_boxes(image, bbox_xyxy, identities, None, names)
                # ........................................................

            print("99999999999999999999999999999", save_single_object)
            # Print time (inference + NMS)
            print(f'{source_path} Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:

                cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
                cv2.imshow("demo", image)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    cv2.destroyAllWindows()
                    raise StopIteration
        frame_number = frame_number + 1
    for _, save_s_o in save_single_object.items():
        save_s_o["video_write"].release()
        if save_s_o["write_count"] < min_number_frame:
            try:
                print("del  ", save_s_o["save_path"])
                os.remove(save_s_o["save_path"])
            except:
                # 报错不做任何处理
                pass


def detect():
    source_dir, save_dir, weights, view_img, save_txt, imgsz, trace = \
        opt.source_dir, opt.save_dir, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    min_number_frame = opt.min_number_frame

    # .... Initialize SORT ....
    # .........................
    # sort_max_age = 25
    # sort_min_hits = 2
    # sort_iou_thresh = 0.2
    # sort_tracker = SortKls(max_age=sort_max_age,
    #                     min_hits=sort_min_hits,
    #                     iou_threshold=sort_iou_thresh)

    # deep_sort = build_tracker(use_cuda=True, age=10)
    # .........................
    # # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    source_sub_dirs = os.listdir(source_dir)
    source_sub_dirs.sort()
    num_sub_dirs = len(source_sub_dirs)
    for i in range(num_sub_dirs):
        sub_dir_path = os.path.join(source_dir, source_sub_dirs[i])
        sub_dir_save = os.path.join(save_dir, source_sub_dirs[i])
        # 子源视频目录
        # 判断是否为路径
        if os.path.isdir(sub_dir_path):
            files = os.listdir(sub_dir_path)
            files.sort()
            num_files = len(files)
            for j in range(num_files):
                file_path = os.path.join(sub_dir_path, files[j])
                # 判断是否为文件
                if os.path.isfile(file_path):
                    file_name, file_extend = os.path.splitext(files[j])
                    if file_extend in [".mp4", ".MP4", ".mkv"]:
                        # if not os.path.exists(sub_dir_save):
                        #     os.makedirs(sub_dir_save)
                        process_single_video(file_path, sub_dir_save, file_name, model, imgsz, device, half,
                                             names, view_img, min_number_frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false')
    parser.add_argument('--source_dir', type=str, required=True, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--save_dir', type=str, required=True, help='save_dir')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--min-number-frame', type=int, default=50, help='min number of frames for generate video )')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='object_tracking', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--colored-trk', action='store_true', help='assign different color to every track')

    parser.set_defaults(download=True)
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))
    if opt.download and not os.path.exists(str(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
