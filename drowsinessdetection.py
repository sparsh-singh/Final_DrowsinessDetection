import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')


lbl=['Close','Open']

model = load_model('models\drowsinessfinalfresh.h5')

path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0
while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces=face.detectMultiScale(gray,1.3,5)




    for (fx,fy,fw,fh) in faces:
        cv2.rectangle(frame, (fx,fy) , (fx+fw,fy+fh) , (0,255,0) , 3 )


        gray_face_img=gray[fy:fy+fh,fx:fx+fw]
        left_eye=leye.detectMultiScale(gray_face_img)
        right_eye=reye.detectMultiScale(gray_face_img)





        for (x,y,w,h) in right_eye:
            r_eye=gray_face_img[y:y+h,x:x+w]
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=np.reshape(r_eye,(1,24,24,1))
            rpred = model.predict_classes(r_eye)
            #print("right",rpred[0])
            break

        for (x,y,w,h) in left_eye:
            l_eye=gray_face_img[y:y+h,x:x+w]
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=np.reshape(l_eye,(1,24,24,1))
            lpred = model.predict_classes(l_eye)
            #print("left",lpred)
            break

        if(rpred[0]==0 and lpred[0]==0):
            score=score+0.3
        else:
            score=max(0,score-0.3)

        cv2.putText(frame,'Closed Score:'+str(int(score)),(100,100), font, 1,(255,255,255),1,cv2.LINE_AA)
        if(score>15):
            cv2.rectangle(frame, (fx,fy) , (fx+fw,fy+fh) , (0,0,255) , 3 )
            try:
                sound.play()

            except:  # isplaying = False
                pass

    cv2.imshow('frame',frame)


    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
