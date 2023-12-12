import time

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt5.QtWidgets import QApplication, QMainWindow, QButtonGroup, QComboBox

from OC_track.ocsort import OCSort
from UI import Ui_MainWindow

from PyQt5 import QtGui
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QDesktopServices

import cv2
import os
import sys
import numpy as np

import yaml

from tools import update_center_points, draw_info
from yolov5_62.tools.inferer import Yolov5


class HyperLink(object):
    '''
    超链接控制
    '''

    def __init__(self, MyMainWindow):
        MyMainWindow.label_csdn.setOpenExternalLinks(True)  # 允许打开外部链接
        MyMainWindow.label_csdn.setCursor(Qt.PointingHandCursor)  # 更改光标样式
        # 设置网址超链接
        url_blog = QUrl("https://blog.csdn.net/qq_28949847/article/details/132412611")
        MyMainWindow.label_csdn.setToolTip(url_blog.toString())
        # 博客点击
        MyMainWindow.label_csdn.mousePressEvent = lambda event: self.open_url(
            url_blog) if event.button() == Qt.LeftButton else None

        MyMainWindow.label_mbd.setOpenExternalLinks(True)  # 允许打开外部链接
        MyMainWindow.label_mbd.setCursor(Qt.PointingHandCursor)  # 更改光标样式
        # 设置网址超链接
        url_mbd = QUrl("https://mbd.pub/o/author-cGeYm2xm/work")
        MyMainWindow.label_mbd.setToolTip(url_blog.toString())
        # 博客点击
        MyMainWindow.label_mbd.mousePressEvent = lambda event: self.open_url(
            url_mbd) if event.button() == Qt.LeftButton else None

        MyMainWindow.label_tb.setOpenExternalLinks(True)  # 允许打开外部链接
        MyMainWindow.label_tb.setCursor(Qt.PointingHandCursor)  # 更改光标样式
        # 设置网址超链接
        url_tb = QUrl("https://mbd.pub/o/author-cGeYm2xm/work")
        MyMainWindow.label_tb.setToolTip(url_blog.toString())
        # 博客点击
        MyMainWindow.label_tb.mousePressEvent = lambda event: self.open_url(
            url_tb) if event.button() == Qt.LeftButton else None

    # 打开超链接
    def open_url(self, url):
        QDesktopServices.openUrl(url)


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        HyperLink(self)
        self.start_type = None
        self.img = None
        self.video = None
        self.video_path = None
        self.cam = 0
        # 绘制了识别信息的frame
        self.img_show = None
        # 是否结束识别的线程
        self.start_end = False
        self.sign = True

        self.worker_thread = None
        self.result_info = None

        self.detect_track = '目标跟踪'
        self.selected_text = '所有目标'
        self.output_dir = './output'
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.comboBox_value = '所有目标'
        # 实现按钮取值
        self.button_group = QButtonGroup()
        self.button_group.addButton(self.radioButton_detect)
        self.button_group.addButton(self.radioButton_track)
        # self.button_group.addButton(radio_button3)
        self.button_group.buttonClicked.connect(self.handle_radio_button_clicked)

        # 打开视频
        self.pushButton_video.clicked.connect(self.open_video)
        # 打开cam
        self.pushButton_cam.clicked.connect(self.open_cam)
        # 绑定开始运行
        self.pushButton_start.clicked.connect(self.start)

        self.ProjectPath = os.getcwd()  # 获取当前工程文件位置

        # 读取YAML文件
        with open(os.path.join(self.ProjectPath, "yolov5_62/data/coco.yaml"), "r", encoding='utf8') as file:
            data = yaml.safe_load(file)
        self.names = data['names']

        self.comboBox.activated.connect(self.onComboBoxActivated)
        self.comboBox.mousePressEvent = self.handle_mouse_press

    def handle_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.sign = False
            # 清空列表
            self.comboBox.clear()
            # new_values = ['New Value 1', 'New Value 2', 'New Value 3']
            self.comboBox.addItems(self.comboBox_value)
        QComboBox.mousePressEvent(self.comboBox, event)

    def onComboBoxActivated(self, index):
        self.sign = True
        self.selected_text = self.comboBox.currentText()

    def handle_radio_button_clicked(self, button):
        self.detect_track = button.text()

    def show_frame(self, img):
        self.update()  # 刷新界面
        if img is not None:
            # 尺寸适配
            size = img.shape
            if size[0] / size[1] > 1.0907:
                w = size[1] * self.label_img.height() / size[0]
                h = self.label_img.height()
            elif size[0] / size[1] < 1.0907:
                w = self.label_img.width()
                h = size[0] * self.label_img.width() / size[1]
            else:
                w, h = self.label_img.width(), self.label_img.height()
            shrink = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_AREA)
            shrink = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)
            QtImg = QtGui.QImage(shrink[:], shrink.shape[1], shrink.shape[0], shrink.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)
            self.label_img.setPixmap(QtGui.QPixmap.fromImage(QtImg))

    def open_video(self):
        try:
            # 选择文件
            self.video_path, filetype = QFileDialog.getOpenFileName(None, "选择文件", self.ProjectPath,
                                                                    "Video Files (*.mp4 *.avi)")
            if self.video_path == "":  # 未选择文件
                self.start_type = None
                return 0
            self.start_type = 'video'
            self.label_video_path.setText(self.video_path)
            self.label_cam_path.setText(" 开启摄像头设备识别")

            cap = cv2.VideoCapture(self.video_path)
            # 读取第一帧
            ret, self.img = cap.read()
            # 显示原图
            self.show_frame(self.img)
        except Exception as e:
            print(e)

    def open_cam(self):
        try:
            # 点击一次打开摄像头，再点击关闭摄像头
            if self.label_cam_path.text() == '已打开摄像头':
                self.label_cam_path.setText(" 开启摄像头设备识别")
                self.start_type = None
            else:
                self.label_cam_path.setText(" 已打开摄像头")
                self.label_video_path.setText(" 选择视频文件")

                self.cam = 0
                self.start_type = 'cam'
                cap = cv2.VideoCapture(self.cam)
                # 读取第一帧
                ret, self.img = cap.read()
                # 显示原图
                self.show_frame(self.img)
        except Exception as e:
            print(e)

    def show_all(self, img, info):
        '''
        展示所有的信息
        '''
        self.show_frame(img)
        self.show_info(info)

    def start(self):
        try:
            if self.start_type == 'video':
                if self.pushButton_start.text() == '开始运行 >':
                    # 开始识别标志
                    self.start_end = False
                    # 开启线程，否则界面会卡死
                    self.worker_thread = WorkerThread(self.video_path, self)
                    self.worker_thread.result_ready.connect(self.show_all)
                    self.worker_thread.start()
                    # 修改文本为结束识别
                    self.pushButton_start.setText("结束运行 >")

                elif self.pushButton_start.text() == '结束运行 >':
                    # 修改文本为开始运行
                    self.pushButton_start.setText("开始运行 >")
                    # 结束识别
                    self.start_end = True

            elif self.start_type == 'cam':
                if self.pushButton_start.text() == '开始运行 >':
                    # 开始识别标志
                    self.start_end = False

                    # 开启线程，否则界面会卡死
                    self.worker_thread = WorkerThread(self.cam, self)
                    self.worker_thread.result_ready.connect(self.show_all)
                    self.worker_thread.start()
                    # 修改文本为结束识别
                    self.pushButton_start.setText("结束运行 >")

                elif self.pushButton_start.text() == '结束运行 >':
                    # 修改文本为开始运行
                    self.pushButton_start.setText("开始运行 >")
                    # 结束识别
                    self.start_end = True

        except Exception as e:
            print(e)

    def add_new_values(self):
        print("adding new values")
        self.comboBox.clear()  # 清空下拉列表值
        new_values = ['New Value 1', 'New Value 2', 'New Value 3']
        for value in new_values:
            self.comboBox.addItem(value)  # 添加新值

    def show_info(self, result):
        try:
            # 显示识别结果
            self.label_time_consum.setText(result['consum_time'])
            self.label_nums.setText(str(result['cls_nums']))
            self.label_classes.setText(str(result['cls_name']))
            self.label_score.setText(str(result['score']))

            self.update()  # 刷新界面
        except Exception as e:
            print(e)


class WorkerThread(QThread):
    '''
    识别视频进程
    '''
    result_ready = pyqtSignal(np.ndarray, dict)

    def __init__(self, path, main_window):
        super().__init__()
        self.path = path
        self.img = None
        self.main_window = main_window

    def run(self):
        try:
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise ValueError("Unable to open video file or cam")
            if self.path == 0:
                self.main_window.img_path = 'camera'
            else:
                self.main_window.img_path = self.path
            # 存放中心点坐标
            dic_center_points = {}
            while True:
                if not self.main_window.sign:
                    time.sleep(1)
                    continue
                result_info = {}
                # 清空所有选项
                ret, self.img = cap.read()
                if not ret or self.main_window.start_type is None or self.main_window.start_end is True:
                    break
                t1 = time.time()
                # try:
                results = y5.infer(self.img, conf_thres=conf_thres, classes=classes)
                tracker_bbox = OCtracker.update(results, self.img.shape)
                # 获取下拉列表值
                self.get_comboBox_value(tracker_bbox)
                if len(tracker_bbox) == 0 or len(results) == 0:
                    continue
                consum_time = str(round(time.time() - t1, 2)) + 's'
                dic_center_points = update_center_points(tracker_bbox, dic_center_points)

                if self.main_window.selected_text == '所有目标':
                    box = results[0][2]
                    score = results[0][1]
                    cls_name = results[0][0]
                else:
                    cls_text, id_text = self.main_window.selected_text.split('_')
                    for bbox in tracker_bbox:
                        box = bbox[:4]
                        id = int(bbox[6])
                        cls_name = bbox[4]
                        score = bbox[5]
                        if cls_text == cls_name and int(id_text) == id:
                            tracker_bbox = [bbox]
                            break
                    else:
                        box = [0, 0, 0, 0]
                        cls_name = '目标丢失'
                        score = 1
                        tracker_bbox = []
                # 画跟踪结果
                self.img_show = draw_info(self.img, tracker_bbox, dic_center_points,
                                          self.main_window.detect_track)

                # 用时
                result_info['consum_time'] = consum_time
                # 目标数
                result_info['cls_nums'] = len(tracker_bbox)
                # 类别
                result_info['cls_name'] = cls_name
                # 置信度
                result_info['score'] = round(score, 2)
                # 位置
                result_info['label_xmin_v'] = int(box[0])
                result_info['label_ymin_v'] = int(box[1])
                result_info['label_xmax_v'] = int(box[2])
                result_info['label_ymax_v'] = int(box[3])
                # 将数据传递回渲染
                self.result_ready.emit(self.img_show, result_info)
        except Exception as e:
            print(e)

    def get_comboBox_value(self, tracker_bbox):
        '''
        获取当前所有的类别和ID，点击下拉列表时，使用
        '''
        # 默认第一个是 所有目标
        lst = ["所有目标"]
        for bbox in tracker_bbox:
            id = int(bbox[6])
            cls_name = bbox[4]
            # preson_1,person_2格式
            lst.append(str(cls_name) + '_' + str(id))
        self.main_window.comboBox_value = lst


if __name__ == "__main__":
    weights = './weights/yolov5x6.pt'
    device = '0'
    conf_thres = 0.4
    classes = None

    y5 = Yolov5(weights=weights, device=device)
    OCtracker = OCSort(det_thresh=0.2, iou_threshold=0.4, delta_t=3, use_byte=False)

    # 创建QApplication实例
    app = QApplication([])
    # 创建自定义的主窗口对象
    window = MyMainWindow()
    # 显示窗口
    window.show()
    # 运行应用程序
    app.exec_()
