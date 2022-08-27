'''
import cv2
import sys

# Get user supplied values
imagePath = 'anh.png'
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
'''
import cv2
import os
cascPathface = "haarcascade_frontalface_default.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)
font=cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
        cv2.putText(frame+str(x),(x,y),font,2,(0,0,255),2,cv2.LINE_AA)
        faceROI = frame[y:y+h,x:x+w]
        # eyes = eyeCascade.detectMultiScale(faceROI)
        # for (x2, y2, w2, h2) in eyes:
        #     eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
        #     radius = int(round((w2 + h2) * 0.25))
        #     frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

        # Display the resulting frame
    cv2.imshow('Video', frame)
    key=cv2.waitKey(1) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()    