from commonfunctions import *



def preprocess(img):        # dy msh hansta5dmha ana 3amlha bs 3ashan a3raf a3ml test
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.float32)/9
    dst = cv2.filter2D(gray,-1,kernel)
    ret,thresh1 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    thresholded_img = thresh1
    return thresholded_img

def get_class_name(class_number):
    classes = ["ZERO", "ONE", "TWO", "THREE","FOUR","FIVE","SIX","SEVEN","EIGHT","NINE","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    return classes[int(class_number)]



def Template_matching(img,template):      # matching 2 images
    count=0
    #print(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j]==template[i,j] and img[i,j]!=255 ):  # don`t count the background
                count+=1
    return count


def Character_Recognition(test_img):     # take the test image and compare it with all characters and letters and
    num_of_matched_pixels = []           # then return the most character having common features with it
 #   test_img = preprocess(test_img)
    #test_img = np.resize(test_img, (300, 250))    # the window size will change depending on the images size
    for filename in sorted(glob.glob('Character_images/*.PNG')):    # after making character segmentation
        img = cv2.imread(filename)
        img = preprocess(img)
        width=test_img.shape[1]
        height=test_img.shape[0]
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img[img>0]=1
    #    print(img.shape,test_img.shape)
       # show_images([img,test_img], ["TemplateImage","Test_img"])
        matched_pixels = Template_matching(img, test_img)
        num_of_matched_pixels.append(matched_pixels)
        print(matched_pixels)
    num_of_matched_pixels = np.array(num_of_matched_pixels)
    index = np.argmax(num_of_matched_pixels)
    print(num_of_matched_pixels)
    #print("class name is : ", get_class_name(index))
    return get_class_name(index)

