import cv2
import pickle
import numpy as np
# video kamera ile aldığımız görüntüleri preProcess etmemiz lazım
def preProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)# histogramı 0-255 arasına genişletme 
    img=img/255
    return img

#########################################################

# Kamera
cap=cv2.VideoCapture(0)
cap.set(3,480)
cap.set(4,480)

#########################################################

# eğittiğimiz modelimizi yükleyelim
pickle_in=open("../output/model_trained_new.p","rb")
model=pickle.load(pickle_in)

# sınıf adlarımızı yükleyelim
with open("../output/class_names.pkl", "rb") as pickle_in:
    class_names = pickle.load(pickle_in)

#########################################################

while True:
    ret,frame=cap.read()
    img=np.asarray(frame)
    img=cv2.resize(img,(32,32))
    img=preProcess(img)
    img=img.reshape(1,32,32,1)
    
    #####################################################
    
    # Predict(Tahmin):
    
    # Tahminler yapılır
    predictions = model.predict(img)

    # Tahmin edilen sınıf indeksi
    classIndex = np.argmax(predictions)

    # Tahmin edilen olasılık değeri
    probVal = np.max(predictions)
    
    if probVal>0.7:
        class_name = class_names[classIndex]
        cv2.putText(frame, f"{class_name}     {probVal:.2f}", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
    
    #####################################################
    
    cv2.imshow("frame",frame)
    if cv2.waitKey(1)&0xff==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()