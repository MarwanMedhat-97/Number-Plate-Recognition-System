from commonfunctions import *
from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
# from imutils import perspective
import numpy as np
# import imutils
import cv2
def Skew_Angle(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.blur(img, (3, 3))  # Blur the image to remove the noise (apply a Gaussian low pass filter 3x3)
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)  #Convert the grayscale image to a binary image

    thresh=np.copy(img)
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    print(thresh)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]  # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        # otherwise, just take the inverse of the angle to make
        # it positive
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    #rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    return rotated


def Segement_Char(img, ShowStep):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #  gray=Skew_Angle(gray)
    #gray=np.copy(img)
    # Get The Thresholding TODO:get another
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.medianBlur(gray, 3)
    GlobalThresh = threshold_otsu(gray)

    ThreshImage = np.copy(gray)
    ThreshImage[gray >= GlobalThresh] = 1
    ThreshImage[gray < GlobalThresh] = 0
    Parition = np.copy(ThreshImage)
    Binary = np.copy(img)
    Parition = cv2.medianBlur(Parition, 3)
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
    #Parition=Skew_Angle(Parition)
   # Parition=Parition
    if ShowStep:
        show_images([Parition], ["Char Segementation FOR THIS PLATE after preprocessing and skew"])
    # -----------------------------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 3))
    dilated = cv2.erode(Parition, kernel, iterations=5)
    if ShowStep:
        show_images([dilated], ["After Dilation"])
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    result = np.copy(img)
    maxArea = -1
    for contour in contours:
        if cv2.contourArea(contour) > maxArea:
            maxArea = cv2.contourArea(contour)
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < maxArea:
            continue
        result = Parition[y:y + h, x:x + w]
        show_images([result], ["Plate Numbers"])

    Parition = result
    Height = np.shape(Parition)[0]
    Wdith = np.shape(Parition)[1]
   # Parition = cv2.medianBlur(Parition, 5)

    #Parition[0:15]=1
    if ShowStep:
        show_images([Parition], ["After median Filter"])
    MAX = 0
    Base_INDEX = 0
    for i in range(Parition.shape[0]):
        max = np.sum(Parition[i])
        if max >= \
                MAX:
            MAX = max
            Base_INDEX = i
    ThreshImage[Base_INDEX, :] = np.ones(ThreshImage.shape[1])
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

    #Parition[MaxTransitionIndex+25:]=1
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
                CurrVP = np.sum(Parition[:, Start + k])
                # print(CurrVP,"wee",StartIndex,EndIndex)
                if CurrVP == 0:
                    if ShowStep:
                        print("Detect Gap in the Word ")
                    #  print(partition[:, StartIndex:EndIndex])
                    CutIndex = k + Start
                    ThereIsGap = True
                    break
            if ThereIsGap:
                ThreshImage[:, CutIndex] = np.zeros(ThreshImage.shape[0])
                if abs(Dummy - CutIndex) < 7:
                    continue  # garbage
                Window = np.copy(Parition[:, Dummy: CutIndex])
                Dummy = CutIndex
                if Dummy == 0:
                    Dummy = CutIndex
                    continue
                Dummy = CutIndex
                CharacterList.append(Window)
            else:
                if abs(Start-End)<10:
                    continue
              #  if(np.sum(Parition[MaxTransitionIndex-20:MaxTransitionIndex+40,CutIndex])):
               #     continue
               # CutIndex=int((Start+End)/2)
               # Window = np.copy(Parition[:, Dummy: CutIndex])
               ## Dummy = CutIndex
               # if Dummy == 0:
                #    Dummy = CutIndex
               #     continue
               # Dummy = CutIndex
               # CharacterList.append(Window)
                continue


    Window = np.copy(Parition[:, Dummy:])
    CharacterList.append(Window)
    if ShowStep:
        show_images([ThreshImage], ["Plate With Cut Segementation"])
    for CharImage in CharacterList:
        if ShowStep:
            show_images([CharImage], ["Char Segementation"])
    return CharacterList
