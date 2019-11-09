from Preprocessing import Preprocess


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


for filename in sorted(glob.glob('../03-Dataset/*.jpg')):
    img = cv2.imread(filename)
    img = Preprocess(img)
    SE=np.ones((5,5))# Use disk SE
    imgOpening = Opening(img,SE) #Opening Morphological
    imgSub=imgOpening-img #Subtraction


    GlobalThresh=0.5#TODO:global threshold level is calculated by using Otsu's method
    imgSub[imgSub>=GlobalThresh]=0
    imgSub[imgSub<GlobalThresh] = 1
    #TODO:Edge Detection by Sobel Operator
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow(filename, imgSub)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
