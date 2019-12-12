from Preprocessing import *
from Character_Segementation import *
from Character_Recognition import *

# TODO: 1- CharacterSegementation by connected components  (Least Significatnt) to get more accurcy  ,2-Char_Recognition get an impleneted module   ,3- implement harris manually and handle the corner case



ShowSteps = 0
FrameList = []
FrameList = extractImages("../03-Dataset/video12.mp4"
                          )
for Frame in FrameList:
    PlateList = []
    #TEST_img = io.imread("../03-Dataset/xx.png")
    PLATE_img, PlateInFrame = Harris(Frame)
    CharacterList = []
    show_images([PLATE_img,PlateInFrame], ["The plate ?","Img with red rectangle ?"])
    # Characters Segementation =>
    CharacterList = Segement_Char(PLATE_img)
    PlateNumber = ""
    # Characters Recognition =>
    for Char in CharacterList:
        #Char=1-Char
        PlateNumber += Character_Recognition(Char)
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
