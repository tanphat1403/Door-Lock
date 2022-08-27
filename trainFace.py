
from imutils import paths
import cv2
import os
from PIL import Image
import numpy as np

path='dataset'
#imagePaths= [os.path.join(path,f) for f in os.listdir(path)]

recognizer=cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# def getImageAndMSSV(path):
#     imagePaths= [os.path.join(path,f) for f in os.listdir(path)]
#     faceSamples=[]
#     MSSVS=[]

#     for imagePath in imagePaths:
#         PIL_img= Image.open(imagePath).convert('L')
#         img_numpy=np.array(PIL_img,'uint8')
#         MSSV=int(os.path.split(imagePath)[-1].split('.')[1])
#         faces=detector.detectMultiScale(img_numpy)

#         for (x,y,w,h) in faces:
#             faceSamples.append(img_numpy[y:y+h,x:x+w])
#             MSSVS.append(MSSV)
#     return faceSamples, MSSVS
def getImageAndMSSV(path):
    imagePaths= [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    MSSVS=[]

    for imagePath in imagePaths:
        images=cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
        MSSV=int(os.path.split(imagePath)[-1].split('.')[1])
        faceSamples.append(np.asarray(images,dtype=np.uint8))
        MSSVS.append(MSSV)
    return faceSamples, MSSVS


print('DOI XU LI...')
faces, MSSVS=getImageAndMSSV(path)
recognizer.train(faces,np.asarray(MSSVS))
recognizer.write('trainer.yml')
print('done')




 