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


def extractImages(pathIn):
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


def Harris(img, ShowSteps):
    # Preprocessing on frame=>
    orginal_img = np.copy(img)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    GrayLevel = np.copy(image)
    image = cv2.bilateralFilter(image, 11, 17, 17)
    # equ = cv2.equalizeHist(image)

    # image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # sat = cv2.medianBlur(image, ksize=3)
    sat = np.copy(image)
    # sat = gaussian_filter(sat , sigma=1)
    GlobalThresh = threshold_otsu(sat)
    ThreshImage = np.copy(sat)
    ThreshImage[sat >= GlobalThresh] = 1
    ThreshImage[sat < GlobalThresh] = 0
    # ThreshImage = cv2.medianBlur(ThreshImage, ksize=3)
    # ThreshImage = GammaCorrection(ThreshImage, 1, 3)
    if ShowSteps:
        show_images([sat, ThreshImage, GrayLevel], ["Image before Harris ", "Binary Image", "gray?"])
    # ----------------------------------------------------------------------------------------
    # print(imgg.shape)
    dst = my_cornerHarris(ThreshImage)
    dst = cv2.morphologyEx(dst, op=cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                           iterations=3)
    if ShowSteps:
        show_images([dst])
    imag = np.copy(img)
    arr = ((dst > 0.1 * dst.max()) & (image > 150))
    if ShowSteps:
        show_images([arr])
    CurrMax = -1
    maxi = [0, 0]
    wx = int(arr.shape[0] / 10.5)
    wy = int(arr.shape[1] / 3)
    ListPlates = []
    # Try those dimensitions ?
    #   plate_dimensions = (
    #  0.03 * label_image.shape[0], 0.08 * label_image.shape[0], 0.15 * label_image.shape[1], 0.3 * label_image.shape[1])
    # plate_dimensions2 = (
    # 0.08 * label_image.shape[0], 0.2 * label_image.shape[0], 0.15 * label_image.shape[1], 0.4 * label_image.shape[1])
    #  print(arr)
    ThreshImage = 1 - ThreshImage
    for i in range(arr.shape[0] - 1, 0 + wx, -int(wx / 3)):  # tool
        # if (i < int(arr.shape[0] / 2)):
        #   break
        for j in range(arr.shape[1] - 1, 0 + wy, -int(wy / 3)):  # 3ard
            # print(np.sum(arr[i-wx:i,j-wy:j]))
            #  print(np.count_nonzero(ThreshImage[i - wx:i, j - wy:j] == 0),np.sum(arr[i - wx:i, j - wy:j]))
            # print(np.count_nonzero(ThreshImage[i - wx:i, j - wy:j] == 0))
            if (np.sum(arr[i - wx:i, j - wy:j])) > CurrMax:  # and np.count_nonzero(
                # ThreshImage[i - wx:i, j - wy:j] != 0) > 5000:

                contours, hierarchy = cv2.findContours(ThreshImage[i - wx:i, j - wy:j], cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                MIN_PIXEL_WIDTH = 1
                MIN_PIXEL_HEIGHT = 2

                MIN_ASPECT_RATIO = 0.25
                MAX_ASPECT_RATIO = 1.0

                MIN_PIXEL_AREA = 10

                #  print(len(contours))
                index = 0
                countValidChar = 0
                # show_images([ThreshImage[i - wx:i, j - wy:j]])
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # print(x, y, w, h, cv2.contourArea(cnt))
                    print(w * h, w, h)
                    if 50 > w * h > MIN_PIXEL_AREA and w > MIN_PIXEL_WIDTH and h > MIN_PIXEL_HEIGHT and MIN_ASPECT_RATIO < float(
                            w) / float(h) and float(w) / float(h) < MAX_ASPECT_RATIO:
                        countValidChar += 1
                print(countValidChar)
            #  if countValidChar < 4:
            #  continue

            CurrMax = np.sum(arr[i - wx:i, j - wy:j])
            maxi[0] = i - wx
            maxi[1] = j - wy
            ListPlates.append([i - wx, j - wy, CurrMax])

    # Preprocessing on The plate =>

    ret = imag[maxi[0]:maxi[0] + wx, maxi[1]:maxi[1] + wy]
    print(maxi[0], maxi[1])
    show_images([imag[maxi[0]:maxi[0] + wx, maxi[1]:maxi[1] + wy]])
    # return ret, imag
    #    imageret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
    imageret = np.copy(ret)
    start, end = 1, 1
    # print(imageret.shape)
    starti, endi = 1, 1
    # show_images([imageret], ["lol"])
    for i in range(imageret.shape[1] - int(0.15 * imageret.shape[1]) - 1):
        if np.average((imageret[int(0.5 * imageret.shape[0]):int(0.8 * imageret.shape[0]),
                       i:i + int(0.15 * imageret.shape[1])])) > 100:
            start = i
        break

    for i in range(imageret.shape[1], int(0.15 * imageret.shape[1]), -1):
        if np.average((imageret[int(0.5 * imageret.shape[0]):int(0.8 * imageret.shape[0]),
                       i - int(0.15 * imageret.shape[1]):i])) > 100:
            end = i
            break
    filtered = imageret[:, start:end]
    # show_images([filtered])
    for i in range(filtered.shape[0] - int(0.3 * filtered.shape[0])):
        if np.average((filtered[i:i + int(0.3 * filtered.shape[0]), :])) > 120:
            starti = i
            break

    for i in range(filtered.shape[0], int(0.3 * filtered.shape[0]), -1):
        if np.average((filtered[i - int(0.3 * filtered.shape[0]):i, :])) > 120:
            endi = i
        break
    # print(starti, endi, "XD")
    filtered = ret[starti:endi, start:end]
    # print(start, end, starti, endi)
    # print(maxi[0], maxi[1], wx, wy)
    cv2.rectangle(img, (start + maxi[1], starti + maxi[0]), (maxi[1] + end, maxi[0] + endi), (255, 0, 0), 4)

    #  print(starti, endi)
    if ShowSteps:
        show_images([ret], ["orginal"])
    # return filtered, 0
    return filtered, img


# print(CurrMax, "Max:")
# if CurrMax == 0:
# print("NO PLATES in this image")
# return ret, img
#   cv2.rectangle(imag, (maxi[1], maxi[0]), (maxi[1] + wy, maxi[0] + wx), (255, 0, 0), 2)

# ListPlates.reverse()
#  if len(ListPlates) == 0:
#    print("NO PLATES in this image")
#   return img, img


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
    image = cv2.bilateralFilter(image, 11, 17, 17)
    # image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    sat = cv2.medianBlur(image, ksize=3)
    show_images([sat], ["After preprocessing"])
    GlobalThresh = threshold_otsu(sat)
    ThreshImage = np.copy(sat)
    ThreshImage[sat >= GlobalThresh] = 1
    ThreshImage[sat < GlobalThresh] = 0
    ThreshImage = cv2.medianBlur(ThreshImage, ksize=3)
    # sat=GammaCorrection(image,1,10)
    show_images([sat, ThreshImage])
    # ----------------------------------------------------------------------------------------
    # print(imgg.shape)
    # dst = cv2.cornerHarris(sat, 20, 5, 0.12)
    dst = my_cornerHarris(sat)
    dst[dst<0.1*dst.max()]=0
    for y in range(int(0.4*dst.shape[0])):
        dst[y,:]=0
    show_images([dst], ["suppose to do this ?"])
    # dst = cv2.dilate(dst, kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    dst = cv2.morphologyEx(dst, op=cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                           iterations=3)
    show_images([dst])
    imag = np.copy(img)
    arr = ((dst > 0.1 * dst.max()) & (image > 150))
    show_images([arr])
    max = -1
    maxi = [0, 0]
    wx = int(arr.shape[0] / 10.5)
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
        return ret, img
    #   cv2.rectangle(imag, (maxi[1], maxi[0]), (maxi[1] + wy, maxi[0] + wx), (255, 0, 0), 2)

    ListPlates.reverse()
    if len(ListPlates) == 0:
        print("NO PLATES in this image")
        return img, img
    #  cv2.rectangle(imag, (ListPlates[1][0], ListPlates[1][1]), (ListPlates[1][0] + wy, ListPlates[1][1] + wx),
    #              (255, 0, 0), 2)
    ret = imag[maxi[0]:maxi[0] + wx, maxi[1]:maxi[1] + wy]
    #  show_images([ret, imag])
    # return ret, imag
    imageret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
    start, end = 1, 1
    # print(imageret.shape)
    starti, endi = 1, 1
    show_images([imageret], ["lol"])
    for i in range(imageret.shape[1] - int(0.15 * imageret.shape[1]) - 1):
        if np.average((imageret[int(0.5 * imageret.shape[0]):int(0.8 * imageret.shape[0]),
                       i:i + int(0.15 * imageret.shape[1])])) > 100:
            start = i
            break
    for i in range(imageret.shape[1], int(0.15 * imageret.shape[1]), -1):
        if np.average((imageret[int(0.5 * imageret.shape[0]):int(0.8 * imageret.shape[0]),
                       i - int(0.15 * imageret.shape[1]):i])) > 100:
            end = i
            break
    filtered = imageret[:, start:end]
    show_images([filtered])
    for i in range(filtered.shape[0] - int(0.3 * filtered.shape[0])):
        if np.average((filtered[i:i + int(0.3 * filtered.shape[0]), :])) > 120:
            starti = i
            break

    for i in range(filtered.shape[0], int(0.3 * filtered.shape[0]), -1):
        if np.average((filtered[i - int(0.3 * filtered.shape[0]):i, :])) > 120:
            endi = i
            break
    print(starti, endi, "XD")
    filtered = ret[starti:endi, start:end]
    print(start, end, starti, endi)
    print(maxi[0], maxi[1], wx, wy)
    cv2.rectangle(img, (start + maxi[1], starti + maxi[0]), (maxi[1] + end, maxi[0] + endi), (255, 0, 0), 4)

    print(starti, endi)
    show_images([ret], ["orginal"])
    # return filtered, 0
    return filtered, img
