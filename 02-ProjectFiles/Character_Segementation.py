from commonfunctions import *
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, skeletonize, thin

def Segement_Char(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    GlobalThresh = threshold_otsu(gray)+50
    ThreshImage = np.copy(gray)
    ThreshImage[gray >= GlobalThresh] = 0
    ThreshImage[gray < GlobalThresh] = 1

    Parition = np.copy(ThreshImage)
    #Parition[Parition.shape[0]-10:Parition.shape[0]]=np.zeros(Parition.shape[1])
    Height = np.shape(Parition)[0]
    Wdith = np.shape(Parition)[1]
    flag = 0
    CharacterList = []
    Parition=1-Parition
    # for h in Height:
    # ASSUMITON => Background is 0 and Characters  are equal to 1
    LastCut=0
    Parition=thin(1-Parition,30)
    Parition=1-Parition
    show_images([Parition], ["Char Segementation"])
    for w in range(Wdith):
        if np.sum(Parition[:, w]) != np.sum(np.ones(Height)) and flag == 0:  # a Char is found
            Start = w
            flag = 1  # loop until the charcter end and found the end of first char
        elif np.sum(Parition[:, w]) == np.sum(np.ones(Height))  and flag == 1:  # char is ended
            End = w
            if abs(End-Start) <15:
                continue # garbage
           # CutIndex = int((Start + End) / 2)
            ThreshImage[:,End+2]=np.ones(Height)
            Window = np.copy(Parition[:,Start: End])
            CharacterList.append(Window)
            flag=0 # Find the next
#            LastCut=CutIndex
    show_images([1 - ThreshImage], ["Plate With Cut Segementation"])
    for CharImage in CharacterList:
        show_images([1 - CharImage], ["Char Segementation"])


img = io.imread("Untitled.png")
  #  img = Preprocess(img)

Segement_Char(img)