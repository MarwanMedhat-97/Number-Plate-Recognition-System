from Preprocessing import Preprocess
from commonfunctions import *


def my_erosion(img, mask):
    shape = img.shape
    new_img = np.copy(img)
    out = int(np.floor(len(mask) / 2))
    for i in range(out, shape[0] - out):
        for j in range(out, shape[1] - out):
            portion = img[i - out:i + out + 1, j - out:j + out + 1]
            mat = np.multiply(mask, portion)
            new_img[i, j] = np.min(mat)
    return new_img


def my_dilation(img, mask):
    shape = img.shape
    new_img = np.copy(img)
    out = int(np.floor(len(mask) / 2))
    for i in range(out, shape[0] - out):
        for j in range(out, shape[1] - out):
            portion = img[i - out:i + out + 1, j - out:j + out + 1]
            mat = np.multiply(mask, portion)
            new_img[i, j] = np.max(mat)
    return new_img


def Opening(mPic, SE):
    return my_dilation(my_erosion(mPic, SE), SE)


def Closing(mPic, SE):
    return my_erosion(my_dilation(mPic, SE), SE)


def Mult_(A, C):
    res = 0
    for i in range(3):
        for j in range(3):
            res += int(A[i][j] * C[i][j])

    return res


def SobelEdgeDetection(img):
    Dim = np.shape(img)
    WindowWidth = 3
    WindowHeight = 3
    EdgeX = (int)(WindowWidth / 2)
    EdgeY = (int)(WindowWidth / 2)
    XEdges = np.zeros((Dim[0], Dim[1]))
    YEdges = np.zeros((Dim[0], Dim[1]))
    EDGE_ = np.zeros((Dim[0], Dim[1]))
    hx = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    hy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    out = int(np.floor(len(hx) / 2))
    for i in range(out, img.shape[0] - out):
        for j in range(out, img.shape[1] - out):
            portion = img[i - out:i + out + 1, j - out:j + out + 1]
            XEdges[i][j] = np.sum(np.multiply(portion, hx))
            YEdges[i][j] = np.sum(np.multiply(portion, hy))
            EDGE_[i][j] = np.sqrt(XEdges[i][j] ** 2 + YEdges[i][j] ** 2)
    return XEdges, YEdges, EDGE_


for filename in sorted(glob.glob('../03-Dataset/*.jpg')): # looping on each image in the folder
    img = cv2.imread(filename)# read the image
    img = Preprocess(img) # output is expected to be GRAYSCALE and no noise
    # Using disk SE
    SE_Size = 50
    SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SE_Size - 1, 2 * SE_Size - 1))
    imgOpening = Opening(img, SE)  # Opening Morphological
    imgSub = img-imgOpening   # Subtraction
    GlobalThresh = threshold_otsu(imgOpening)# global threshold level is calculated by using Otsu's method
    imgThresh=np.copy(imgSub)
    imgThresh[imgSub >= GlobalThresh] = 1
    imgThresh[imgSub < GlobalThresh] = 0
    show_images([img, imgOpening, imgSub,imgThresh], ["Orignal Image ", "After Opening ", "Img Subtruction","Image Binariziation"])
    #Edge Detection by Sobel Operator
    ImgX, ImgEdgeY, ImgEdge = SobelEdgeDetection(imgThresh)
    show_images([img, imgOpening, imgSub, imgThresh,ImgEdge],
                ["Orignal Image ", "After Opening ", "Img Subtruction", "Image Binariziation","Edge Detection"])
    SE_Close=np.ones((3,3))
    imgClosing = Closing(ImgEdge, SE)  # Opening Morphological
    #show_images([img, imgOpening, imgSub, imgThresh, ImgEdge,imgClosing],
         #       ["Orignal Image ", "After Opening ", "Img Subtruction", "Image Binariziation", "Edge Detection","Filling Holes"])