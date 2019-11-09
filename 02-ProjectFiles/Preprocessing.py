from commonfunctions import *
def Preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, (3, 3))# TODO: Write this Function
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)# TODO: Write this Function
    return img

for filename in sorted(glob.glob('../03-Dataset/*.jpg')):
    img = cv2.imread(filename)
    img = Preprocess(img)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow(filename, img)
    cv2.waitKey(0)