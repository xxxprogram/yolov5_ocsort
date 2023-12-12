import torch
import numpy as np
import cv2
import time

from pathlib import Path
import sys
import os

# cur_path = os.path.abspath(os.path.dirname(__file__))
# yolov5_path = os.path.abspath(os.path.dirname(cur_path))


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # YOLOv5 root directory
# ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from yolov5_62.models.common import DetectMultiBackend
from yolov5_62.utils.augmentations import letterbox

from yolov5_62.utils.general import (check_img_size, non_max_suppression, scale_boxes)
from yolov5_62.utils.plots import Annotator, colors, save_one_box
from yolov5_62.utils.torch_utils import select_device, time_sync

coco_yaml = os.path.join(ROOT, "data", "coco.yaml")
yolo5s_weights = os.path.join(ROOT, "weights", "yolov5s.pt")


class Yolov5:

    def __init__(self, weights=yolo5s_weights, device=0, yaml=coco_yaml, imgsz=(640, 640), half=False, dnn=False):

        # load model
        device = select_device(device)

        if not torch.cuda.is_available():
            device = torch.device("cpu")

        self.device = device
        model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=yaml, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        bs = 1
        self.model = model
        self.stride = stride
        self.names = names
        self.imgsz = imgsz
        self.half = half
        self.pt = pt
        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    def precess_image(self, img_src, img_size, stride, half, auto=True):
        '''Process image before image inference.'''
        # Padded resize
        img = letterbox(img_src, img_size, stride=stride, auto=self.pt)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255  # 0 - 255 to 0.0 - 1.0
        return img

    def infer(self, img_src, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=100):
        """
        yolov5 image infer \n
        :param img_src: source image numpy format
        :param conf_thres: Confidence Threshold
        :param iou_thres:   IOU Threshold
        :param classes: classes
        :return results: detection results: list [['person', 0.95, [3393, 811, 3836, 1417]], ...] 左上角、右下角
        """

        img = self.precess_image(img_src, self.imgsz, self.stride, self.half)

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        pred = self.model(img, augment=False, visualize=False)
        det = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

        # 格式转换
        lst_result = []
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img_src.shape).round()
            results = det.cpu().detach().numpy()
            # 格式转换
            for detection in results:
                index = detection[5]  # 获取类别索引
                cls_name = self.names.get(index, 'unknown')  # 根据索引获取类别名称，未知类别则设置为'unknown'
                new_detection = [cls_name, detection[4], [detection[0], detection[1], detection[2], detection[3]], [], [], [], []]  # 构建新的检测结果项
                lst_result.append(new_detection)

        return lst_result
