import numpy as np
import cv2
import skimage.io as io
from skimage.color import rgb2gray
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.ndimage import gaussian_filter
from commonfunctions import *


def Preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (3, 3))
    # _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    return img


def SobelFilter(img):
    image = rgb2gray(img)
    sobelMaskXDir = np.array([[-1, 0, 1],
                              [-np.sqrt(2), 0, np.sqrt(2)],
                              [-1, 0, 1]])
    sobelMaskYDir = sobelMaskXDir.T
    Gx = convolve2d(image, sobelMaskXDir)
    Gy = convolve2d(image, sobelMaskYDir)
    G = np.sqrt(np.power(Gx, 2))
    return Gx


def MorhOperations(img):
    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 16))
    SE2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    SE3 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    x = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=SE)

    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel=SE2, iterations=1)
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel=SE3, iterations=1)
    return x


def extractImages(pathIn,IsRotated):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    success = True
    ListFrames = []
    while success:
        # save a frame for every second
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
        success, image = vidcap.read()
        #   print('Read a new frame: ', success)
        if success == 0:
            break
        if IsRotated:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        #   cv2.ipmwrite("../03-Dataset/frames/"+"frame%d.jpg" %
        #          count, image)     # save frame as JPEG file
        ListFrames.append(image)
        count = count + 1
    return ListFrames


def GammaCorrection(image, c, gamma):
    #   image = rgb2gray(image)
    imageGamma = np.array(image)
    image = c * np.power(imageGamma, gamma)
    return image

def my_cornerHarris(Orignal_img):
    img = rgb2gray(Orignal_img)
    # Sobel Algorithm =>
    fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    I_x = sig.convolve2d(img, fx, mode='same')
    I_y = sig.convolve2d(img, fy, mode='same')
    Ixx = gaussian_filter(I_x ** 2, sigma=4)
    Ixy = gaussian_filter(I_y * I_x, sigma=4)
    Iyy = gaussian_filter(I_y ** 2, sigma=4)
    k = 0.05
    detA = Ixx * Iyy - Ixy ** 2
    traceA = Ixx + Iyy
    harris_response = detA - k * traceA ** 2
    return harris_response


def Working_Harris(img, Steps):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   # image=GammaCorrection(image,1.1,2)
    image = cv2.bilateralFilter(image, 11, 17, 17)
    # image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    sat = cv2.medianBlur(image, ksize=3)
    show_images([sat], ["After preprocessing"])
    if Steps:
        show_images([sat], ["After preprocessing"])
    GlobalThresh = threshold_otsu(sat)
    ThreshImage = np.copy(sat)
    ThreshImage[sat >= GlobalThresh] = 1
    ThreshImage[sat < GlobalThresh] = 0
    ThreshImage = cv2.medianBlur(ThreshImage, ksize=3)
    #sat=GammaCorrection(image,1,10)
    #show_images([sat, ThreshImage])
    # dst = cv2.cornerHarris(sat, 20, 5, 0.12)
    dst = my_cornerHarris(sat)
    dst[dst<0.1*dst.max()]=0
    for y in range(int(0.4*dst.shape[0])):
        dst[y,:]=0
    dst = cv2.morphologyEx(dst, op=cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                           iterations=3)
    imag = np.copy(img)
    arr = ((dst > 0.1 * dst.max()) & (image > 150))
    max = -1
    maxi = [0, 0]
    wx = int(arr.shape[0] / 11.5)
    wy = int(arr.shape[1] / 3)
    ListPlates = []
    #  print(arr)
    for i in range(arr.shape[0] - 1, 0, -int(wx / 3)):  # tool
        if (i < int(arr.shape[0]*0.4)):
            break
        for j in range(arr.shape[1] - 1, 0 + wy, -int(wy / 3)):  # 3ard
            # print(np.sum(arr[i-wx:i,j-wy:j]))
            if (np.sum(arr[i - wx:i, j - wy:j])) >= max and (
                    np.count_nonzero(ThreshImage[i - wx:i, j - wy:j] == 0) > 20):
                max = np.sum(arr[i - wx:i, j - wy:j])
                maxi[0] = i - wx
                maxi[1] = j - wy
                ListPlates.append([i - wx, j - wy, max])

    # print(max, "Max:")
    if max == 0:
        print("NO PLATES in this image")
        return img, img

    ListPlates.reverse()
    if len(ListPlates) == 0:
        print("NO PLATES in this image")
        return img, img
    ret = imag[maxi[0]:maxi[0] + wx, maxi[1]:maxi[1] + wy]
    imageret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
    if Steps:
        show_images([imageret], ["lol"])
    ret, new_img = cv2.threshold(imageret, 150, 255, cv2.THRESH_BINARY_INV)  # for black text , cv.THRESH_BINARY_INV
    if Steps:
        show_images([new_img], ["After Thresholding"])

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4,3))
    dilated = cv2.erode(new_img, kernel, iterations=5)
    dilated=255-dilated
    if Steps:
        show_images([dilated], ["After Dilation"])
    contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    result=np.copy(img)
    maxArea=-1
    for contour in contours:
        if cv2.contourArea(contour)>maxArea:
            maxArea=cv2.contourArea(contour)
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < maxArea:
            continue
        cv2.rectangle(img, (x+ maxi[1], y+ maxi[0]), (x + w+ maxi[1], y + h+ maxi[0]), (255, 0, 0), 4)
        result=imag[y+ maxi[0]:y+ maxi[0]+h,x+ maxi[1]:x+ maxi[1]+w]
        if Steps:
            show_images([img],["Bounding Box(s)"])



    return result, img
