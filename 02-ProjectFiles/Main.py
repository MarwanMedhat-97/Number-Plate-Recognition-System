from Plate_Detection import *
from Character_Segementation import *
from Character_Recognition import *






ShowSteps = 0
FrameList = []
VideoName = "VideoCar.mp4"
Dest = "../03-Dataset/" + VideoName
FrameList = extractImages(Dest)
PlateFrameList = []
for Frame in FrameList:
    PlateList = []
    # TEST_img = io.imread("../03-Dataset/xx.png")
    PLATE_img, PlateInFrame = Working_Harris(Frame, 0)
    CharacterList = []
    show_images([PLATE_img, PlateInFrame], ["The plate ?", "Img with red rectangle ?"])
    # Characters Segementation =>
    CharacterList = Segement_Char(PLATE_img, ShowSteps)
    PlateNumber = ""
    # Characters Recognition =>
   # for Char in CharacterList:
    #    PlateNumber += Character_Recognition(Char, ShowSteps)
    PlateFrameList.append(PlateInFrame)
    print(PlateNumber)

height, width, layers = PlateFrameList[0].shape
size = (width, height)
VIDEO_NAME = VideoName[:len(VideoName)-4]+"PlateDetection.avi"
out = cv2.VideoWriter(VIDEO_NAME, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for Frame in PlateFrameList:
    out.write(Frame)
out.release()
