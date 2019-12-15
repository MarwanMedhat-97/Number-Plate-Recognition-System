from commonfunctions import *
import pytesseract
from scipy import ndimage


def character_recognition(test_img):     # take the test image and compare it with all characters and letters and
    '''
    num_of_matched_pixels = []           # then return the most character having common features with it
#    test_img = preprocess(test_img)
    test_img[test_img > 0] = 1
    #test_img = np.resize(test_img, (300, 250))    # the window size will change depending on the images size
    for filename in sorted(glob.glob('Character_images/alphabet/*.PNG')):    # after making character segmentation
        img = cv2.imread(filename)
        img = preprocess(img)
        width=test_img.shape[1]
        height=test_img.shape[0]
        dim = (int(14*width/15), int(4*height/6)+1)

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img[img>0]=1
        #print(img.shape,test_img.shape)

        matched_pixels = Template_matching(test_img[int(height/6):int(5*height/6),int(width/15)+1:width],img)
        num_of_matched_pixels.append(matched_pixels)
        #print(matched_pixels)
    num_of_matched_pixels = np.array(num_of_matched_pixels)
    index = np.argmax(num_of_matched_pixels)
   # print(num_of_matched_pixels)
    #print("class name is : ", get_class_name(index))
    return get_class_name(index)
    '''
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    # converts the image to result and saves it into result variable
    result = pytesseract.image_to_string(test_img)
    # write text in a text file and save it to source path
    return result