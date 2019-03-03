#coding=utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
import cv2
import os


def read_file(first_path='C:\\Users\\Yunqing\\Desktop\\dataset\\new\\'):
    # read all added features files iteratively and process into new csv file
    img_list = os.listdir(first_path)
    i = 0
    for img_file in img_list:
        i += 1
        img = cv2.imread(first_path+img_file, 0)
        img_info = np.array(img).shape
        image_height = img_info[0]
        image_weight = img_info[1]
        dst = np.zeros((image_height, image_weight, 1), np.uint8)
        for i in range(image_height):
            for j in range(image_weight):
                grayPixel = img[i][j]
                dst[i][j] = 255 - grayPixel
        cv2.imwrite('C:\\Users\\Yunqing\\Desktop\\dataset\\la\\'+img_file, dst)


read_file()