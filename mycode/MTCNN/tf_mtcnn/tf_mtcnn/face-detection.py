# 把一些警告的訊息暫時関掉
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import random
from tqdm import tqdm
from scipy import misc
import numpy
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import detect_face

minsize = 20 # 最小的臉部的大小
threshold = [ 0.6, 0.7, 0.7 ]  # 三個網絡(P-Net, R-Net, O-Net)的閥值
factor = 0.709 # scale factor

gpu_memory_fraction=0.0
print('Creating networks and loading parameters')

with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        #GPU
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        #CPU
        #sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}, log_device_placement=True))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None) # 構建三個網絡(P-Net, R-Net, O-Net)

# 注意: OpenCV讀進來的圖像所產生的Numpy Ndaary格式是BGR (B:Blue, G: Green, R: Red) 
# 跟使用PIL或skimage的格式RGB (R: Red, G: Green, B:Blue)在色階(channel)的順序上有所不同
image = cv2.imread('oscar.jpg')
bounding_boxes, points = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
faces_detected = len(bounding_boxes)
print('Total faces detected ：{}'.format(faces_detected))
for face_position in bounding_boxes:
    face_position=face_position.astype(int)
    cv2.rectangle(image, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
    #crop=img[face_position[1]:face_position[3],face_position[0]:face_position[2],]
    #x1 = face_position[0] if face_position[0] > 0 else 0
    #y1 = face_position[1] if face_position[1] > 0 else 0
    #x2 = face_position[2] if face_position[2] > 0 else 0
    #y2 = face_position[3] if face_position[3] > 0 else 0 
    #cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('face',image)
cv2.imwrite('face.jpg',image)
cv2.waitKey(0)

