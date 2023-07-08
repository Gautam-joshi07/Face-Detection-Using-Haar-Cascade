import cv2

        # make a function draw boundary for bounding box near face/ detecting face
def draw_boundary(img, classifier, scaleFactor,minNeighbor,color,text):
    # convert image into gray scale
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(grayImg,scaleFactor,minNeighbor)
    coords = []

    for (x,y,w,h) in features:

        # draw bounding box for detecting face
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        # it will put the text above the bounding box on x axis
        cv2.putText(img,text,(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        coords = [x,y,w,h]

    return coords

    #  make a function detect for detecting a face and eyes
def detect(img,facecascade,eyecascade):

    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0)}
    # calling draw boundary function to make a bounding box near face
    # detecting face
    coords = draw_boundary(img, facecascade, 1.1, 4, color['red'], "Face")

    if len(coords) == 4:
        #detecting eyes
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords [0]+coords[2]]
        coords = draw_boundary(roi_img, eyecascade, 1.1, 4, color['red'], "eye")

    return img


# add haar cascades file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


# take input video from camera
video = cv2.VideoCapture(0)

# for detecting images
# img = cv2.imread("lena_image.jpg")

while True:
    # read input video
    _, img = video.read() # comment this line when detecting images
    # calling detect function
    img = detect(img,face_cascade,eye_cascade)
    cv2.imshow("Face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q for quiting the live streaming
        break

video.release()
cv2.destroyAllWindows()