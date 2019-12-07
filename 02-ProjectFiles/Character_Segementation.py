from commonfunctions import *
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
#from imutils import perspective
import numpy as np
#import imutils
import cv2
def Segement_Char(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    GlobalThresh = threshold_otsu(gray)
    ThreshImage = np.copy(gray)
    ThreshImage[gray >= GlobalThresh] = 1
    ThreshImage[gray < GlobalThresh] = 0
    Parition = np.copy(ThreshImage)
    Binary = np.copy(Parition)
    # Parition[Parition.shape[0]-10:Parition.shape[0]]=np.zeros(Parition.shape[1])
    Height = np.shape(Parition)[0]
    Wdith = np.shape(Parition)[1]
    flag = 0
    CharacterList = []
    # Parition=1-Parition
    # for h in Height:
    # ASSUMITON => Background is 0 and Characters  are equal to 1
    LastCut = 0
   # Parition = skeletonize(1 - Parition)
   # Parition = 1 - Parition
    show_images([Parition], ["Char Segementation FOR THIS PLATE"])
    #GETING MAX transition row
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
    flag=0
    Parition=1-Parition
    Dummy=0
    for w in range(Wdith):
        if np.sum(Parition[MaxTransitionIndex, w]) ==0 and flag == 0:  # a Char is found
            Start = w
            flag = 1  # loop until the charcter end and found the end of first char
        elif np.sum(Parition[MaxTransitionIndex, w]) == 1 and flag == 1:  # char is ended
            End = w
            flag=0
            ThereIsGap = False
            for k in range(abs(Start - End) + 1):
                CurrVP = np.sum(Parition[MaxTransitionIndex-10:MaxTransitionIndex+10, Start + k])
                # print(CurrVP,"wee",StartIndex,EndIndex)
                if CurrVP == 0:
                    print("Detect Gap in the Word ")
                    #  print(partition[:, StartIndex:EndIndex])
                    CutIndex = k + Start
                    ThereIsGap = True
                    break
            if ThereIsGap:
                ThreshImage[:, CutIndex] = np.zeros(Height)
                Window = np.copy(ThreshImage[MaxTransitionIndex-10:MaxTransitionIndex+10, Dummy: CutIndex])
                Dummy=CutIndex
                if Dummy ==0:
                    Dummy=CutIndex
                    continue
                Dummy=CutIndex
                CharacterList.append(Window)
            else:
                continue

            # if abs(End-Start) <20:
            #    continue # garbage
         #   CutIndex = int((Start + End) / 2)

         #   flag = 0  # Find the next
    #            LastCut=CutIndex
    Window = np.copy(ThreshImage[MaxTransitionIndex-10:MaxTransitionIndex+10, Dummy: ])
    CharacterList.append(Window)
    show_images([ThreshImage], ["Plate With Cut Segementation"])
    for CharImage in CharacterList:
        show_images([CharImage], ["Char Segementation"])
    return CharacterList


def Segement_Char_2(img):
    V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 29, offset=15, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    thresh = cv2.bitwise_not(thresh)

    # resize the license plate region to a canonical size
#    plate = cv2.resize(img, width=400)
 #   thresh = cv2.resize(thresh, width=400)
    cv2.imshow("Thresh", thresh)

    labels = measure.label(thresh, neighbors=8, background=0)
    charCandidates = np.zeros(thresh.shape, dtype="uint8")
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask to display only connected components for the
        # current label, then find contours in the label mask
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    if len(cnts) > 0:
        # grab the largest contour which corresponds to the component in the mask, then
        # grab the bounding box for the contour
        c = max(cnts[0], key=cv2.contourArea)
        (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

        # compute the aspect ratio, solidity, and height ratio for the component
        aspectRatio = boxW / float(boxH)
        solidity = cv2.contourArea(c) / float(boxW * boxH)
        heightRatio = boxH / float(img.shape[0])

        # determine if the aspect ratio, solidity, and height of the contour pass
        # the rules tests
        keepAspectRatio = aspectRatio < 1.0
        keepSolidity = solidity > 0.15
        keepHeight = heightRatio > 0.4 and heightRatio < 0.95

        # check to see if the component passes all the tests
        if keepAspectRatio and keepSolidity and keepHeight:
            # compute the convex hull of the contour and draw it on the character
            # candidates mask
            hull = cv2.convexHull(c)
            cv2.drawContours(charCandidates, [hull], -1, 255, -1)

# img = io.imread("Untitled.png")
#  img = Preprocess(img)

# List=[]
# List =Segement_Char(img)
# str=""
# for img in List:
#    str=str+(Character_Recognition(img))
# print (str)
