from Plate_Detection import *
from Character_Segementation import *
from Character_Recognition import *
#from Character_Identification import *

# TODO: 1- CharacterSegementation by connected components  (Least Significatnt) to get more accurcy  ,2-Char_Recognition get an impleneted module   ,3- implement harris manually and handle the corner case








ShowSteps = 0
FrameList = []
VideoName = "car6.mp4"
Dest = "../03-Dataset/" + VideoName
FrameList = extractImages(Dest)
PlateFrameList = []
for Frame in FrameList:
    PlateList = []
    # TEST_img = io.imread("../03-Dataset/xx.png")
    PLATE_img, PlateInFrame = Working_Harris(Frame, 1)
    CharacterList = []
    show_images([PLATE_img, PlateInFrame], ["The plate ?", "Img with red rectangle ?"])
    # Characters Segementation =>
    CharacterList = Segement_Char(PLATE_img, ShowSteps)
    PlateNumber = ""
    # Characters Recognition =>
    for Char in CharacterList:
        PlateNumber += Character_Recognition(char,ShowSteps)
    PlateFrameList.append(PlateInFrame)
    print(PlateNumber)

    # Display the resulting frame
#    show_images([PlateInFrame])
#   cv2.imshow('Frame', PlateInFrame)
# Press Q on keyboard to  exit
#  if cv2.waitKey(25) & 0xFF == ord('q'):
#     break

# height, width, layers = frame.shape

# Output Video =>
# video = cv2.VideoWriter(video_name, 0, 1, (width,height))

# for image in images:
#   video.write(cv2.imread(os.path.join(image_folder, image)))
height, width, layers = PlateFrameList[0].shape
size = (width, height)
VIDEO_NAME = VideoName[:len(VideoName)-4]+"PlateDetection.avi"
out = cv2.VideoWriter(VIDEO_NAME, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for Frame in PlateFrameList:
    out.write(Frame)
out.release()
