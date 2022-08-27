from imutils import paths
import cv2
import os
from PIL import Image
import numpy as np
import sqlite3
from datetime import datetime
import time


#Luu lai lich su mo khoa
def update_history(MSSV,name,date,his_path): 
    sql="INSERT INTO history (MSSV,name,date,image) VALUES ("+str(MSSV)+',"'+name+'","'+str(date)+'","'+his_path+'")'
    conn.execute(sql)
    conn.commit()
# Them hoac xoa anh o dataset
def Insert_Delete_data(MSSV,name,flag): #them hoac xoa anh o data set, flag=1 la insert, flag=0 la xoa 
    count=0
    if(flag==0):
        imagePaths= [os.path.join('dataset',f) for f in os.listdir('dataset')]
        for imagepath in imagePaths:
            if str(MSSV) in imagepath:
                os.remove(imagepath)
    else:
        while(True):
            ret, frame=cap.read()
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray,
                                                scaleFactor=1.3,
                                                minNeighbors=5,
                                                minSize=(60, 60),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
                                            
            for (x,y,w,h) in faces:
                face=frame[y:y+h,x:x+w]
                face=cv2.resize(face,(200,200))
                gray=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                #cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
                cv2.imwrite('dataset/'+ name +'.'+str(MSSV)+'.'+str(count)+'.jpg', gray)
                count+=1
                time.sleep(5)
                print(count)
                #cv2.imshow("frame",frame)
            #if not os.path.exists('dataset'):
            # os.makedirs('dataset')
            cv2.waitKey(1)
            if count>3:
                break
        # cap.release()
        # cv2.destroyAllWindows()

def train_data():
    path='dataset'
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # def getImageAndMSSV(path):
    imagePaths= [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    MSSVS=[]

    for imagePath in imagePaths:
        images=cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
        MSSV=int(os.path.split(imagePath)[-1].split('.')[1])
        faceSamples.append(np.asarray(images,dtype=np.uint8))
        MSSVS.append(MSSV)
    


    print('DOI XU LI...')
# faces, MSSVS=getImageAndMSSV(path)
    recognizer.train(faceSamples,np.asarray(MSSVS))
    # recognizer.write('trainer.yml')
    print('done')

    # def getImageAndMSSV(path):
    #     imagePaths= [os.path.join(path,f) for f in os.listdir(path)]
    #     faceSamples=[]
    #     MSSVs=[]

    #     for imagePath in imagePaths:
    #         PIL_img= Image.open(imagePath).convert('L')
    #         img_numpy=np.array(PIL_img,'uint8')
    #         MSSV=int(os.path.split(imagePath)[-1].split('.')[1])
    #         faces=face_cascade.detectMultiScale(img_numpy)

    #         for (x,y,w,h) in faces:
    #             faceSamples.append(img_numpy[y:y+h,x:x+w])
    #             MSSVs.append(MSSV)
    #     return faceSamples, MSSVs
    # print('DOI XU LI...')
    # faces, MSSVs=getImageAndMSSV(path)
    # recognizer.train(faces,np.array(MSSVs))
    # recognizer.write('trainer.yml')

def getProfile(MSSV): #lay thong tin MSSV va name tu csdl
    query="SELECT * FROM people where MSSV="+str(MSSV)
    cursor=conn.execute(query)
    conn.commit()
    profile=None
    for row in cursor:
        profile=row
    return profile




conn=sqlite3.connect('C:/xampp/htdocs/demo/mydb.db')
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()   
cursor=conn.execute("SELECT MSSV,name FROM people")
prev_num_rows=cursor.fetchall()
# recognizer.read('trainer.yml')
cap = cv2.VideoCapture(0)
font=cv2.FONT_HERSHEY_SIMPLEX
train_data()
print('bat dau')
dem=0
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                            scaleFactor=1.1,
                                            minNeighbors=5,
                                            )
    for (x,y,w,h) in faces:
        face=frame[y:y+h,x:x+w]
        face=cv2.resize(face,(200,200))
        # roiGray=gray[y:y+h,x:x+w]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        _MSSV, conf=recognizer.predict(gray)
        if conf < 60:
            dem=0
            profile=getProfile(_MSSV)
            print(_MSSV)
            date=datetime.now() 
            imagePaths= [os.path.join('C:/xampp/htdocs/demo/history_image',f) for f in os.listdir('C:/xampp/htdocs/demo/history_image')]
            nums=len(imagePaths)
            his_path='./history_image/'+profile[1] +'.'+str(profile[0])+'.'+str(nums)+'.jpg'
            cv2.imwrite('C:/xampp/htdocs/demo/history_image/'+profile[1] +'.'+str(profile[0])+'.'+str(nums)+'.jpg',frame)
            update_history(profile[0],profile[1],date,his_path)
            
            
            print(str(profile[0])+' vao')  
            print('doi 5s')        
            time.sleep(5)
            print('Xong')
            # cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
            # cv2.putText(frame,profile[1]+str(conf),(x,y),font,2,(0,0,255),2,cv2.LINE_AA)
            
        else:
            dem=dem+1
            if dem==250: 
                print("Phat hien nguoi la")
                dem=0

            # cv2.rectangle(frame, (x, y), (x + w, y + h),(0,255,0), 2)
            # cv2.putText(frame,'Unknown',(x,y),font,2,(0,0,255),2,cv2.LINE_AA)
    # cv2.imshow('frame',frame) 
    # key=cv2.waitKey(1) 
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
     

    #detect add/update or delete 
    cursor=conn.execute("SELECT MSSV,name FROM people")
    num_rows=cursor.fetchall()
    if(num_rows!=prev_num_rows):

        diff=list(set(prev_num_rows) ^ set(num_rows))
        if(len(prev_num_rows)>len(num_rows)): #co nguoi da bi xoa
            Insert_Delete_data( diff[0][0], diff[0][1], 0)
            #delete anh trong dataset
            
        elif(len(prev_num_rows)<len(num_rows)):
            Insert_Delete_data( diff[0][0], diff[0][1], 1)
            #them anh vao dataset
        print('da them')
        prev_num_rows=num_rows
        train_data()



