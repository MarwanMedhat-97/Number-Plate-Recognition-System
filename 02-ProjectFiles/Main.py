from Plate_Detection import *
from Character_Segementation import *
from Character_Recognition import *
from Character_Identification import *
# from Character_Identification import *

ShowSteps = 0
FrameList = []
VideoName = "video15.mp4"
IsRotated = True
Dest = "../03-Dataset/" + VideoName
FrameList = extractImages(Dest, IsRotated)
PlateFrameList = []
for Frame in FrameList:

    # TEST_img = io.imread("../03-Dataset/xx.png") # in case you want to test your image
    PlateList = []
    PLATE_img, PlateInFrame = Working_Harris(Frame, ShowSteps)
    CharacterList = []
    show_images([PLATE_img, PlateInFrame], ["The plate ?", "Img with red rectangle ?"])
    # Characters Segementation =>
    CharacterList = Segement_Char(PLATE_img, ShowSteps)
    PlateNumber = ""
    # Characters Recognition =>
    for Char in CharacterList:
        PlateNumber += character_recognition(Char)
    PlateFrameList.append(PlateInFrame)
    print(PlateNumber)
# Writing Video =>
height, width, layers = PlateFrameList[0].shape
size = (width, height)
VIDEO_NAME = VideoName[:len(VideoName) - 4] + "PlateDetection.avi"
out = cv2.VideoWriter(VIDEO_NAME, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for Frame in PlateFrameList:
    out.write(Frame)
out.release()
