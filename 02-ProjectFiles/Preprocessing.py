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
    if titles is None: titles = ['(%d)' % i for i in range(1, n_ims + 1)]
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


img = io.imread("img.jpg")
x1 = SobelFilter(img)
print(np.min(x1), np.max(x1))
f, thresh = cv2.threshold(x1, thresh=1.2, maxval=255, type=cv2.THRESH_BINARY)

# x= cv2.equalizeHist(x)

x = MorhOperations(thresh)
show_images([img, x])
