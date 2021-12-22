#================================================================
#
#   File name   : detect_mnist.py
#   Author      : PyLessons
#   Created date: 2020-08-12
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : mnist object detection example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import *

video_path = '/home/alexandr/tests/WhatsApp Video 2021-11-12 at 15.29.20.mp4'


yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
#yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use keras weights
yolo.load_weights(f"./checkpoints/yolov4_custom_Tiny")

##    detect_image(yolo, image_path, "mnist_test.jpg", input_size=YOLO_INPUT_SIZE,
##                 show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
detect_video(yolo, video_path, './IMAGES/detected.mp4', input_size=YOLO_INPUT_SIZE,
             show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

    
