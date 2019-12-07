from Preprocessing import *
from Character_Segementation import *
from Character_Recognition import *

ShowSteps = 0
FrameList=[]
FrameList=extractImages("../03-Dataset/EgyRoad.mp4")
for Frame in FrameList:
    PlateList=[]
    img = io.imread("../03-Dataset/lol4.jpg")
    IMG,PlateInFrame=Harris(img)
    CharacterList=[]
    show_images([IMG],["The plate ?"])
    CharacterList=Segement_Char(IMG)
    PlateNumber=""
    for Char in CharacterList:
        PlateNumber+=Character_Recognition(Char)
    print(PlateNumber)

    # Display the resulting frame
#    show_images([PlateInFrame])
 #   cv2.imshow('Frame', PlateInFrame)
    # Press Q on keyboard to  exit
  #  if cv2.waitKey(25) & 0xFF == ord('q'):
   #     break

#height, width, layers = frame.shape

#video = cv2.VideoWriter(video_name, 0, 1, (width,height))

#for image in images:
 #   video.write(cv2.imread(os.path.join(image_folder, image)))


