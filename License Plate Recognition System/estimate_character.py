import sys
import os
import time
import random
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import Image, ImageFont, ImageDraw
import pygame
INPUT_SIZE = 784
WIDTH = 28
HEIGHT = 28
NUM_CLASSES = 34

SAVER_DIR = "python/train-saver/digits/"

LETTERS_DIGITS = ("0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z")
license_num = ""
time_begin = time.time()
# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')
    # 定义全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)

pygame.init()

if __name__ =='__main__':
    start = time.time()
    saver = tf.train.import_meta_graph("%smodel.ckpt.meta"%(SAVER_DIR))
    with tf.Session() as sess:
        model_file=tf.train.latest_checkpoint(SAVER_DIR)
        saver.restore(sess, model_file)

        # 第一个卷积层
        W_conv1 = sess.graph.get_tensor_by_name("W_conv1:0")
        b_conv1 = sess.graph.get_tensor_by_name("b_conv1:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')
        # 第二个卷积层
        W_conv2 = sess.graph.get_tensor_by_name("W_conv2:0")
        b_conv2 = sess.graph.get_tensor_by_name("b_conv2:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')
        # 全连接层
        W_fc1 = sess.graph.get_tensor_by_name("W_fc1:0")
        b_fc1 = sess.graph.get_tensor_by_name("b_fc1:0")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 7 * 7 * 64])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)
        # dropout
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        # readout层
        W_fc2 = sess.graph.get_tensor_by_name("W_fc2:0")
        b_fc2 = sess.graph.get_tensor_by_name("b_fc2:0")
        # 定义优化器和训练op
        conv_result = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        for n in range(2,8):
            path = "python/zxytest_images/%s.jpg" % (n)
            nowimage = Image.open(path)
            nowimage = nowimage.convert('1')
            width = nowimage.size[0]
            height = nowimage.size[1]
            img_input = [[0]*INPUT_SIZE for i in range(1)]
            for h in range(0, height):
                for w in range(0, width):
                    if nowimage.getpixel((w, h)) < 190:
                        img_input[0][w+h*width] = 1
                    else:
                        img_input[0][w+h*width] = 0
            result = sess.run(conv_result, feed_dict = {x: np.array(img_input), keep_prob: 1.0})
            maxx1 = 0
            maxx1_index = 0
            for i in range(NUM_CLASSES):
                if result[0][i] > maxx1:
                    maxx1 = result[0][i]
                    maxx1_index = i
            license_num = license_num + LETTERS_DIGITS[maxx1_index]
    y = license_num
    text = "其它字符识别结果为:%s"%y
    im = Image.new("RGB", (300, 300), (255, 255, 255))
    font = pygame.font.Font(os.path.join("python/fonts/", "msyh.ttc"), 34)
    rtext = font.render(text, True, (0, 0, 0), (255, 255, 255))
    pygame.image.save(rtext, "python/result/3.jpg")
    mat = cv2.imread("python/result/3.jpg")
    cv2.namedWindow("now")
    cv2.moveWindow("now", 430, 400)
    cv2.imshow("now", mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    writer=tf.summary.FileWriter("logs/",sess.graph)
