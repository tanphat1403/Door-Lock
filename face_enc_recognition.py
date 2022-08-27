 
import cv2
import os
import sqlite3

from numpy import int16

def getName(MSSV):
    conn=sqlite3.connect('C:/xampp/htdocs/demo/mydb.db')
    query="SELECT * FROM people where MSSV="+str(MSSV)
    cursor=conn.execute(query)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile

recognizer=cv2.face.LBPHFaceRecognizer_create()
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer.read('trainer.yml')
video_capture = cv2.VideoCapture(0)
video_capture.set(3,640)
video_capture.set(4,480)
# minW=0.1*video_capture.get(3)
# minH=0.1*video_capture.get(4)
font=cv2.FONT_HERSHEY_SIMPLEX
#dict_name={1910429:'PHAT',1111111:'Mr X',2222222:'Mr Y'}
while(True):
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                                minNeighbors=5,
                                                # minSize=(int(minW),int(minH)),
                                        )
    for (x,y,w,h) in faces:
        face=frame[y:y+h,x:x+w]
        face=cv2.resize(face,(200,200))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        _MSSV, conf=recognizer.predict(gray)
        if conf <60:
            profile=getName(_MSSV)
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
            #cv2.putText(frame,profile[1]+str(conf),(x,y),font,2,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(frame,profile[1]+str(x),(x,y),font,2,(0,0,255),2,cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
            cv2.putText(frame,'Unknown',(x,y),font,2,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('frame',frame) 
    key=cv2.waitKey(1) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()     










'''
#find path of xml file containing haarcascade file 
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())
print("Streaming started")
video_capture = cv2.VideoCapture(0)
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE) 
    # convert the input frame from BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
       #Compare encodings with encodings in data["encodings"]
       #Matches contain array with boolean values and True for the embeddings it matches closely
       #and False for rest
        matches = face_recognition.compare_faces(data["encodings"],
         encoding)
        #set name =inknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                #Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
            name = max(counts, key=counts.get)
            
 
 
        # update the list of names
        names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
'''