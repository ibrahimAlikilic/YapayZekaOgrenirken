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

#########################################################

while True:
    ret,frame=cap.read()
    # frame=cv2.flip(frame,1) # normalde olması gerekir lakin kağıda yazığ ekrana gösterdiğimden ters gözüküyor
    img=np.asarray(frame) # frame imi np dizisine dönüştürdüm.
    img=cv2.resize(img,(32,32)) # eğitimimizde görüntülerimizi 32ye 32 olarak kaydetmiştik o yüzden bu dönüşümü gerçekleştiriyoruz
    img=preProcess(img)
    img=img.reshape(1,32,32,1)
    """
    1: Bu, tek bir örneği temsil eder. Modelin tek bir örneği işleyebilmesi için, verinin bu boyutta olması gerekir.
    32x32: Görüntünün boyutları; burada 32x32 piksel.
    1: Görüntüde tek bir renk kanalı olduğunu belirtir (gri tonlamalı bir görüntü).
    """
    
    #####################################################
    
    # Predict(Tahmin):
    
    # Tahminler yapılır
    predictions = model.predict(img)

    # Tahmin edilen sınıf indeksi
    classIndex = np.argmax(predictions)

    # Tahmin edilen olasılık değeri
    probVal = np.max(predictions)
    """
    1. Tahminlerin Yapılması

predictions = model.predict(img)

    model.predict(img): Bu satır, modelinizin img üzerindeki tahminleri yapmasını sağlar. img tek bir görüntü olduğu için, 
    modeliniz bu görüntü için olasılıkları tahmin eder.

    Çıktı: predictions, modelinizin her sınıf için tahmin ettiği olasılıkları içeren bir dizidir. 
    Eğer modeliniz n sınıfı tahmin ediyorsa, predictions dizisi boyutunda (1, n) olacaktır. İlk boyut 1 çünkü tek bir görüntü üzerinde tahmin yapıyoruz, 
    ikinci boyut ise n sınıf için tahmin edilen olasılıkları içerir.

2. Tahmin Edilen Sınıfın İndeksi

classIndex = np.argmax(predictions)

    np.argmax(predictions): predictions dizisindeki en yüksek değeri bulur ve bu değerin indeksini döndürür. 
    Bu indeks, modelinizin tahmin ettiği en olası sınıfı temsil eder.

    Çıktı: classIndex, modelin tahmin ettiği sınıfın indeksini içerir. 
    Örneğin, eğer predictions dizisinde en yüksek değer 0.85 ile 2 indeksinde bulunuyorsa, classIndex 2 olur. 
    Bu, modelin en yüksek olasılığı 2 numaralı sınıfa verdiğini gösterir.

3. Tahmin Edilen Olasılık Değeri

probVal = np.max(predictions)

    np.max(predictions): predictions dizisindeki en yüksek değeri döndürür. Bu değer, modelin tahmin ettiği sınıf için verdiği olasılıktır.

    Çıktı: probVal, tahmin edilen sınıfın olasılığını içerir. Örneğin, eğer predictions dizisinde en yüksek değer 0.85 ise, probVal 0.85 olacaktır. 
    Bu, modelin tahmin ettiği sınıfın ne kadar güvenilir olduğunu gösterir.
    """
    if probVal>0.7:
        cv2.putText(frame,str(classIndex)+"     "+str(probVal),(50,50),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255))
    cv2.imshow("frame",frame)
    if cv2.waitKey(1)&0xff==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()