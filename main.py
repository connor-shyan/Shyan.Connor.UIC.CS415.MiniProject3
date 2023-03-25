#
# CS 415 - Mini Project 3
# Connor Shyan
# UIC, Fall 2022
# Using Segmentation code tutorial as the base
#

import numpy as np
import cv2
import math

#
# Calculate Hue and Saturation Histogram Function (from code tutorial)
#
def calculate_hs_histogram(img, bin_size):
    height, width, _ = img.shape
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    max_h = 179
    max_s = 255
    hs_hist = np.zeros((math.ceil((max_h+1)/bin_size), math.ceil((max_s+1)/bin_size)))
    for i in range(height):
        for j in range(width):
            h = img_hsv[i, j, 0]
            s = img_hsv[i, j, 1]
            hs_hist[math.floor(h/bin_size), math.floor(s/bin_size)] += 1
    hs_hist /= hs_hist.sum()
    return hs_hist

#
# Color Segmentation Function (from code tutorial)
#
def color_segmentation(img, hs_hist, bin_size, threshold):
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((height, width, 1))
    for i in range(height):
        for j in range(width):
            h = hsv[i, j, 0]
            s = hsv[i, j, 1]
            if hs_hist[math.floor(h/bin_size), math.floor(s/bin_size)] > threshold:
                mask[i, j, 0] = 1
    return mask

#
# Convolution Function (from code tutorials)
#
def convolution(im, kernel):
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    im_height, im_width = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size - 1) / 2)
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2))
    im_padded[pad_size:-pad_size, pad_size:-pad_size] = im

    im_out = np.zeros_like(im)
    for x in range(im_width):
        for y in range(im_height):
            im_patch = im_padded[y:y + kernel_size, x:x + kernel_size]
            new_value = np.sum(kernel * im_patch)
            im_out[y, x] = new_value
    return im_out

#
# Function to get Gaussian Kernel (from code tutorials)
#
def get_gaussian_kernel(kernel_size, sigma):
    kernel_x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    for i in range(kernel_size):
        kernel_x[i] = np.exp(-(kernel_x[i] / sigma) ** 2 / 2)
    kernel = np.outer(kernel_x.T, kernel_x.T)

    kernel *= 1.0 / kernel.sum()
    return kernel

#
# Training data from skin patches
#
bin_size = 20
hs_hist = calculate_hs_histogram(cv2.imread("skinpatch0.jpg"), bin_size)
for i in range(1, 10):
    hs_hist += calculate_hs_histogram(cv2.imread("skinpatch" + str(i) + ".jpg"), bin_size)
hs_hist /= 10

#
# Testing for P1
#
img_test = cv2.imread("testing_image.bmp")
threshold = 0.03
mask = color_segmentation(img_test, hs_hist, bin_size, threshold)
img_seg = img_test * mask
cv2.imwrite("p1_testing_mask.png", (mask*255).astype(np.uint8))
cv2.imwrite("p1_testing_segmentation.png", img_seg.astype(np.uint8))
# cv2.imshow("Input", img_test)
# cv2.imshow("Mask", (mask*255).astype(np.uint8))
# cv2.imshow("Segmentation", img_seg.astype(np.uint8))
# cv2.waitKey()

#
# Building Gaussian-based skin color model for P2
#
gaussian_kernel = get_gaussian_kernel(9, 3)
gaussian_data = []
gaussian_diagonal = np.empty([1,1])
for i in range(0, 10):
    gaussian_data.append(convolution(cv2.imread("skinpatch" + str(i) + ".jpg"), gaussian_kernel))
    gaussian_diagonal = np.hstack(gaussian_diagonal, gaussian_data[i].diagonal())
flat_gaussian = gaussian_diagonal.flatten()
mean = np.mean(flat_gaussian)
cov = np.cov(flat_gaussian)

#
# Self-Study Harris Corner Detector for P3
# Reference:https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
#
checkerboard = cv2.imread("checkerboard.png")
grayCheckerboard = cv2.cvtColor(checkerboard, cv2.COLOR_BGR2GRAY)
grayCheckerboard = np.float32(grayCheckerboard)
dstCheckerboard = cv2.cornerHarris(grayCheckerboard, 2, 3, 0.04)
dstCheckerboard = cv2.dilate(dstCheckerboard, None)
checkerboard[dstCheckerboard > 0.01 * dstCheckerboard.max()] = [0, 0, 255]
cv2.imwrite("p3_cv2_harris_checkerboard.png", checkerboard)

toy = cv2.imread("toy.png")
grayToy = cv2.cvtColor(toy, cv2.COLOR_BGR2GRAY)
grayToy = np.float32(grayToy)
dstToy = cv2.cornerHarris(grayToy, 2, 5, 0.07)
dstToy = cv2.dilate(dstToy, None)
toy[dstToy > 0.01 * dstToy.max()] = [0, 0, 255]
cv2.imwrite("p3_cv2_harris_toy.png", toy)
