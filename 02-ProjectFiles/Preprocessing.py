import numpy as np
import cv2
import skimage.io as io
from skimage.color import rgb2gray
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


def show_images(images, titles=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def Preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (3, 3))  # TODO: Write this Function
    # _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)# TODO: Write this Function
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
    while success:
        # save a frame for every second
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite("../03-Dataset/frames/" + "frame%d.jpg" %
                    count, image)  # save frame as JPEG file
        count = count + 1


def Harris(img):
    image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sat = cv2.medianBlur(image, ksize=3)
    dst = cv2.cornerHarris(sat, 20, 5, 0.12)
    # dst = cv2.dilate(dst, kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    dst = cv2.morphologyEx(dst, op=cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                           iterations=3)

    imag = np.copy(img)
    imag1 = np.copy(img)

    imag2 = np.copy(img)
    arr = ((dst > 0.1 * dst.max()) & (image > 150))

    max = -1
    maxi = [0, 0]
    wx = int(arr.shape[0] / 10.5)
    wy = int(arr.shape[1] / 3)
    for i in range(arr.shape[0] - 1, 0, -int(wx / 3)):  # tool
        if (i < int(arr.shape[0] / 2)):
            break
        for j in range(arr.shape[1] - 1, 0 + wy, -int(wy / 3)):  # 3ard
            # print(np.sum(arr[i-wx:i,j-wy:j]))
            if (np.sum(arr[i - wx:i, j - wy:j])) > max:
                max = np.sum(arr[i - wx:i, j - wy:j])
                maxi[0] = i - wx
                maxi[1] = j - wy
    # imag[maxi[0]:maxi[0]+wx,maxi[1]:maxi[1]+wy,:] = [0, 0, 255]
    # cv2.rectangle(imag, (maxi[1], maxi[0]),(maxi[1] + wy, maxi[0] + wx), (255, 0, 0), 2)
    ret = imag[maxi[0]:maxi[0] + wx, maxi[1]:maxi[1] + wy]
    show_images([ret])
    imageret = cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
    start, end = 1, 1
    print(imageret.shape)
    starti,endi=1,1

    for i in range(imageret.shape[1] - 5):
        if np.average((imageret[int(0.2 * imageret.shape[0]):int(0.6 * imageret.shape[0]), i:i + 4])) > 100:
            start = i
            break
    for i in range(imageret.shape[1] - 6, 4, -1):
        if np.average((imageret[int(0.2 * imageret.shape[0]):int(0.6 * imageret.shape[0]), i:i + 4])) > 100:
            end = i
            break
    filtered = imageret[:, start:end]

    for i in range(filtered.shape[1] - 3):
        if np.average((filtered[i:i + 3, :])) > 120:
            starti = i
            break

    for i in range(filtered.shape[0] - 3, 0, -1):
        if np.average((filtered[i:i + 3, :])) > 120:
            endi = i
            break
    filtered=ret[starti:endi, start:end]
    print(starti,endi)

    return filtered


img = io.imread("lol.jpg")
x1 = Harris(img)
show_images([x1])

img = io.imread("lol2.jpg")
x1 = Harris(img)
show_images([x1])

img = io.imread("lol3.jpg")
x1 = Harris(img)
show_images([x1])

# x= cv2.equalizeHist(x)
