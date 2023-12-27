

import argparse
from functools import partial
from pathlib import Path
import numpy as np

import torch
import cv2 
from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from examples.detectors import get_yolo_inferer

#from examples.detectors.custom import CustomTracker





__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from examples.utils import write_mot_results



# Define the polygon vertices
#polygon = np.array([[50, 50], [200, 50], [200, 200], [50, 200]])
# Draw the polygon on a black image
#img = np.zeros((300, 300, 3), dtype=np.uint8)
#cv2.fillPoly(img, [polygon], (255, 255, 255))
# Define the center of a detected object
#cx, cy = 150, 150
# Check if the center is inside the polygon
#dist = cv2.pointPolygonTest(polygon, (cx, cy), False)
#if dist >= 0:
 #   print('Inside')
#else:
 #   print('Outside')
    


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):

    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))


    # if 'yolov8' not in str(args.yolo_model):
    #     # replace yolov8 model
    #     yolo = CustomTracker('custom.pt')

    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args

   
    polygon = np.array([[50, 50], [1000, 50], [1000, 600], [50, 600]])
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 3
    thickness = 3
    org= (100,100)

    for frame_idx, r in enumerate(results):
        # draw the polygon
        color=(0,255,0)
        orig_frame = r.orig_img
        h, w, _ = orig_frame.shape
        print('h', h)
        print('w', w)
        
        # area_1 = [(w-1415, h-100),(w-385, h-100),(w-600, h-500),(w-1280, h-500)]
        cv2.polylines(orig_frame, [np.array(polygon, np.int32)], True, color, 10)

         # Display the image with the polygon
        #cv2.imshow('Polygon', orig_frame)
        cv2.putText(orig_frame, str('name'),(200,1000), font, 
                        fontScale, color, thickness, cv2.LINE_AA)


        if r.boxes.data.shape[1] == 7:

            if yolo.predictor.source_type.webcam or args.source.endswith(VID_FORMATS):
                p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                yolo.predictor.mot_txt_path = p
            elif 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                p = yolo.predictor.save_dir / 'mot' / (Path(args.source).parent.name + '.txt')
                yolo.predictor.mot_txt_path = p

            if args.save_mot:
                write_mot_results(
                    yolo.predictor.mot_txt_path,
                    r,
                    frame_idx,
                )

            if args.save_id_crops:
                for d in r.boxes:
                    print('args.save_id_crops', d.data)
                    save_one_box(
                        d.xyxy,
                        r.orig_img.copy(),
                        file=(
                            yolo.predictor.save_dir / 'crops' /
                            str(int(d.cls.cpu().numpy().item())) /
                            str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                        ),
                         
                    BGR=True
                )
  
        # Modify the code here
        # Initialize the tracked dictionary
        tracked = {}
        # Initialize the counted set
        counted = set()
        # Initialize the flag
        left_roi = False
        # Loop over the detected objects
        for obj in r.boxes:
            # Get the class name and id of the object
            class_name = obj.data[0, 0].item()
            obj_id =    obj.data[0, 1]
            x_min, y_min, x_max, y_max = obj.data[0, 2:6]
           # print('class', class_name)
           #print('id', obj_id)

            # Calculate the center coordinates
            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)
            # Check if the center is inside the rectangle
            dist = cv2.pointPolygonTest(polygon, (cx, cy), False)
            if  dist >= 0:
                # If the object is inside the rectangle, add it to the tracked dictionary
                tracked[obj_id] = class_name
                # Reset the flag
                left_roi = False
            else:
                # If the object is outside the rectangle, check if it was tracked before
                if obj_id in tracked:
                   # If the object was tracked before, add it to the counted set
                   counted.add((obj_id, tracked[obj_id]))
                   # Remove it from the tracked dictionary
                   del tracked[obj_id]
                   # Set the flag
                   left_roi = True
        # If the flag is True, print the counts
        if left_roi:
          print('Counts:', len(counted))                     

    if args.save_mot:
        print(f'MOT results saved to {yolo.predictor.mot_txt_path}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--vid_stride', default=1, type=int,
                        help='video frame-rate stride')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
