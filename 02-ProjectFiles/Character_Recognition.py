from commonfunctions import *

from scipy import ndimage

def preprocess(img):        # dy msh hansta5dmha ana 3amlha bs 3ashan a3raf a3ml test
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(gray,-1,kernel)
    ret,thresh1 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    thresholded_img = thresh1
    return thresholded_img



def Count_connected_parts(
        img):  # this function returns the number of connected parts given the binary image of any letter
    labeled, nr_objects = ndimage.label(
        img < 1)  # 100 is the threshold but in case of binary image given (0,1) it will change
    # print(nr_objects)
    # print("Number of objects is {}".format(nr_objects))
    return nr_objects


def count_holes(img, num_connected_parts):  # count number of holes in each character
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = np.copy(img)
    # kernel = np.ones((3,3),np.float32)/9
    # dst = cv2.filter2D(gray,-1,kernel)
    # print(img)
    # ret,thresh1 = cv2.threshold(img,50,255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print("y= ",len(contours)-1-num_connected_parts)
    return len(contours) - num_connected_parts  # -1 is the contour of the image frame




def Max_transition_rows(img):
    MaxTransition = 0
    # MaxTransitionIndex = Base_INDEX
    for i in range(img.shape[0]):  # loop on Each row
        CurrTransitionRow = 0
        flag = 1
        for j in range(img.shape[1]):  # loop on coloumns for specific row
            if flag == 1 and img[i, j] == 0:
                flag = 0
                CurrTransitionRow += 1
            elif flag == 0 and img[i, j] == 1:
                flag = 1
                CurrTransitionRow += 1

        if CurrTransitionRow >= MaxTransition:
            MaxTransition = CurrTransitionRow
    return MaxTransition


def Max_transition_colomns(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, img = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    MaxTransition = 0
    # MaxTransitionIndex = Base_INDEX
    for i in range(img.shape[1]):  # loop on each coloumn
        CurrTransitionCol = 0
        flag = 1
        for j in range(img.shape[0]):  # loop on rows for specific coloumn
            if flag == 1 and img[j, i] == 0:
                flag = 0
                CurrTransitionCol += 1
            elif flag == 0 and img[j, i] == 1:
                flag = 1
                CurrTransitionCol += 1

        if CurrTransitionCol >= MaxTransition:
            MaxTransition = CurrTransitionCol
    return MaxTransition





def get_class_name(class_number):
    classes = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    return classes[int(class_number)]



def Template_matching(img,template):      # matching 2 images
    img[img > 1] = 1
    #SE = np.ones((3, 3))
    #img = Closing(img, SE)
    #img = img.astype('uint8')

    #show_images([template, img], ["TemplateImage", "Test_img"])
    #num_parts = Count_connected_parts(img)
    #holes = count_holes(img, num_parts)
    #vertical_trans = Max_transition_colomns(img)
    #horizontal_trans = Max_transition_rows(img)
    #print(holes,vertical_trans,horizontal_trans)
    #num_parts2 = Count_connected_parts(template)
    #holes2 = count_holes(template, num_parts2)
    #vertical_trans2 = Max_transition_colomns(template)
    #horizontal_trans2 = Max_transition_rows(template)
    #print(holes2, vertical_trans2, horizontal_trans2)
    count = 0
    img_template_probability_match = 0
    #if(holes==holes2-1 and vertical_trans==vertical_trans2 and horizontal_trans==horizontal_trans2):
        #img = 1 - img
        #template = 1 - template
        #first_image_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        #second_image_hist = cv2.calcHist([template], [0], None, [256], [0, 256])
        #img_template_probability_match = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)[0][0]
        #return img_template_probability_match






    #img = thin(img, 5)
    #template = thin(template, 5)
    #print(img.shape,template.shape)
   # patch = img[i - 1:i + 2, j - 1:j + 2]
        #show_images([1-template,1-img], ["TemplateImage", "Test_img"])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            #patch1 = img[i-2 : i+3 , j-2 : j+3]
            #patch2 = template[i - 2: i + 3, j - 2: j + 3]
            if(img[i,j]==template[i,j] ):  # don`t count the background
                count+=1
    return count


def Character_Recognition(test_img,showSteps):     # take the test image and compare it with all characters and letters and
    num_of_matched_pixels = []           # then return the most character having common features with it
#    test_img = preprocess(test_img)
    test_img[test_img > 0] = 1
    #test_img = np.resize(test_img, (300, 250))    # the window size will change depending on the images size
    for filename in sorted(glob.glob('Character_images/alphabet/*.PNG')):    # after making character segmentation
        img = cv2.imread(filename)
        img = preprocess(img)
        width=test_img.shape[1]
        height=test_img.shape[0]
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img[img>0]=1
        #print(img.shape,test_img.shape)

        matched_pixels = Template_matching(test_img,img)
        num_of_matched_pixels.append(matched_pixels)
        #print(matched_pixels)
    num_of_matched_pixels = np.array(num_of_matched_pixels)
    index = np.argmax(num_of_matched_pixels)
   # print(num_of_matched_pixels)
    #print("class name is : ", get_class_name(index))
    return get_class_name(index)

#img=io.imread('Character_images/K.png')
#str=Character_Recognition(img)
#print(str)