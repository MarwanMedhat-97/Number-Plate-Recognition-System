from Preprocessing import *
from Character_Segementation import *
from Character_Recognition import *

ShowSteps = 0
FrameList = []
FrameList = extractImages("../03-Dataset/car6.mp4")
for Frame in FrameList:
    PlateList = []
    TEST_img = io.imread("../03-Dataset/lol3.jpg")
    my_cornerHarris(Frame)
    PLATE_img, PlateInFrame = Harris(Frame)
    # SE=np.ones((3,3))
    # IMG=median(IMG,SE)
    CharacterList = []
    show_images([PLATE_img], ["The plate ?"])
    # Characters Segementation =>
    CharacterList = Segement_Char(PLATE_img)
    PlateNumber = ""
    # Characters Recognition =>
    for Char in CharacterList:
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
