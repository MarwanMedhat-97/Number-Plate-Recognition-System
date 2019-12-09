from commonfunctions import *
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
# from imutils import perspective
import numpy as np
# import imutils
import cv2


def Segement_Char(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Get The Thresholding TODO:get another
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)
   # blur = cv2.GaussianBlur(img, (5, 5), 0)
    #ret3, th3 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    GlobalThresh = threshold_otsu(gray)

    ThreshImage = np.copy(gray)
    ThreshImage[gray >= GlobalThresh] = 1
    ThreshImage[gray < GlobalThresh] = 0
    Parition = np.copy(ThreshImage)
    Binary = np.copy(img)
    Parition = cv2.medianBlur(Parition, 3)
    show_images([Parition], ["Char Seweqweqwegementation FOR THIS PLATE"])
    # Parition[Parition.shape[0]-10:Parition.shape[0]]=np.zeros(Parition.shape[1])
    Height = np.shape(Parition)[0]
    Wdith = np.shape(Parition)[1]
    flag = 0
    CharacterList = []
    # Parition=1-Parition
    # for h in Height:
    # ASSUMITON => Background is 0 and Characters  are equal to 1
    LastCut = 0
    #Parition = skeletonize(1 - Parition)
    #Parition = 1 - Parition
    show_images([Parition], ["Char Segementation FOR THIS PLATE"])
    #-----------------------------------------------------------------------------

    MAX = 0
    Base_INDEX = 0
    for i in range(Parition.shape[0]):
        max = np.sum(Parition[i])
        if max >= \
                MAX:
            MAX = max
            Base_INDEX = i
    ThreshImage[Base_INDEX,:]=np.ones(ThreshImage.shape[1])
    print(Base_INDEX)

    # GETING MAX transition row
    MaxTransition = 0
    MaxTransitionIndex = 0
    for i in range(0, Height, 1):  # loop on Each row
        CurrTransitionRow = 0
        flag = 0
        for j in range(Wdith):  # loop on coloumns for specific row
            if flag == 0 and Parition[i, j] == 1:
                flag = 1
                CurrTransitionRow += 1
            elif flag == 1 and Parition[i, j] == 0:
                flag = 0

        if CurrTransitionRow >= MaxTransition:
            MaxTransitionIndex = i
            MaxTransition = CurrTransitionRow
    flag = 0
    Parition = 1 - Parition
    Dummy = 0
    for w in range(Wdith):
        if np.sum(Parition[MaxTransitionIndex, w]) == 0 and flag == 0:  # a Char is found
            Start = w
            flag = 1  # loop until the charcter end and found the end of first char
        elif np.sum(Parition[MaxTransitionIndex, w]) == 1 and flag == 1:  # char is ended
            End = w
            flag = 0
            ThereIsGap = False


            for k in range(abs(Start - End) + 1):
                CurrVP = np.sum(Parition[MaxTransitionIndex -30:MaxTransitionIndex + 80, Start + k])
                # print(CurrVP,"wee",StartIndex,EndIndex)
                if CurrVP == 0:
                    print("Detect Gap in the Word ")
                    #  print(partition[:, StartIndex:EndIndex])
                    CutIndex = k + Start
                    ThereIsGap = True
                    break
            if ThereIsGap:
                ThreshImage[:, CutIndex] = np.zeros(Height)
                if abs(Dummy - CutIndex) < 7:
                    continue  # garbage
                Window = np.copy(ThreshImage[:, Dummy: CutIndex])
                Dummy = CutIndex
                if Dummy == 0:
                    Dummy = CutIndex
                    continue
                Dummy = CutIndex
                CharacterList.append(Window)
            else:
                continue

        #   CutIndex = int((Start + End) / 2)

    Window = np.copy(ThreshImage[:, Dummy:])
    CharacterList.append(Window)
    show_images([ThreshImage], ["Plate With Cut Segementation"])
    i=0
    for CharImage in CharacterList:
        cv2.imwrite("../03-Dataset/frames/" + "image%d.jpg" %i, CharImage)
        i+=1
        show_images([CharImage], ["Char Segementation"])
    return CharacterList


