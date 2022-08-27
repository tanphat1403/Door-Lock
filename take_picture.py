from imutils import paths
import cv2
import os
import sqlite3
import time



def Insert_Delete_data(MSSV,name,flag): #them hoac xoa anh o data set, flag=1 la insert, flag=0 la xoa 
    count=0
    conn=sqlite3.connect('C:/xampp/htdocs/demo/mydb.db')
    if flag=='0':       
        query="DELETE FROM people where MSSV="+str(MSSV)
        conn.execute(query)
        imagePaths= [os.path.join('dataset',f) for f in os.listdir('dataset')]
        for imagepath in imagePaths:
            if str(MSSV) in imagepath:
                os.remove(imagepath)
    elif flag=='1':
            cap=cv2.VideoCapture(0)
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
                    
                    print(count)
                    if count==5: print('Done')
                    #cv2.imshow("frame",frame)
                #if not os.path.exists('dataset'):
                # os.makedirs('dataset')
                cv2.waitKey(1)
                if count>4:
                    break
            query="INSERT INTO people(MSSV,name) VALUES("+str(MSSV)+",'"+str(name)+"')"
            conn.execute(query)
    conn.commit()
    conn.close()
        # cap.release()
        # cv2.destroyAllWindows()

# def insertorUpdate(MSSV,name):
#     conn=sqlite3.connect('C:/xampp/htdocs/demo/mydb.db')
#     query="SELECT * FROM people where MSSV="+str(MSSV)
#     cursor=conn.execute(query)
#     isRecorExist=0
#     for row in cursor:
#         isRecorExist=1
#     if (isRecorExist==0):
#         query="INSERT INTO people(MSSV,name) VALUES("+str(MSSV)+",'"+str(name)+"')"
#     else:
#         query="UPDATE people SET name='"+str(name)+"' where MSSV="+str(MSSV)
#     conn.execute(query)
#     conn.commit()
#     conn.close()



#count=0
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# cap=cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
MSSV=input('nhap MSSV: ')
name=input('nhap name: ')
flag=input('Nhan 0 de xoa, 1 de them: ')
Insert_Delete_data(MSSV,name,flag)
#insertorUpdate(MSSV,name)
# while(True):
#     ret, frame=cap.read()
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray,
#                                          scaleFactor=1.3,
#                                          minNeighbors=5,
#                                          minSize=(60, 60),
#                                          flags=cv2.CASCADE_SCALE_IMAGE)
                                    
#     for (x,y,w,h) in faces:
#         face=frame[y:y+h,x:x+w]
#         face=cv2.resize(face,(200,200))
#         gray=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#         cv2.imwrite('dataset/'str(name)+'.'+str(MSSV)+'.'+str(count)+'.jpg', gray)
#         count+=1
#         cv2.imshow("frame",frame)
#         #time.sleep(5)
#     #if not os.path.exists('dataset'):
#         #os.makedirs('dataset')
#     cv2.waitKey(1)
#     if count>4:
# #         break
# cap.release()
# cv2.destroyAllWindows()  
#Insert_Delete_data(MSSV,name,flag) #them hoac xoa anh o data set, flag=1 la insert, flag=0 la xoa 




 