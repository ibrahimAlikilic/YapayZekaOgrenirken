import cv2
import pickle
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array


image=cv2.imread("../input/mnist(1).png") 

#####################################################

print("Selective Search aşaması başladı ...")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()
rects=ss.process()
print("Selective Search aşaması bitti.")

#####################################################

proposals=[]
boxes=[]
output=image.copy()
for (x,y,w,h) in rects[:20]:
    color=[random.randint(0,255)for j in range(0,3)]
    roi=image[y:y+h,x:x+w]
    roi=cv2.resize(roi,dsize=(32,32),interpolation=cv2.INTER_LANCZOS4)
    """
    cv2.INTER_LANCZOS4 nedir?

    Lanczos Enterpolasyonu (Lanczos-4): Bu yöntem, bir sinc fonksiyonuna dayanan yüksek kaliteli bir enterpolasyon algoritması kullanır (Lanczos çekirdeği). INTER_LANCZOS4'deki 4, çekirdekte kullanılan lob sayısını ifade eder ve bu, enterpolasyonun pürüzsüzlüğünü ve doğruluğunu etkiler.

cv2.INTER_LANCZOS4 ne zaman kullanılır?

    Yüksek kaliteli yeniden boyutlandırma: INTER_LANCZOS4, bir görüntüyü yeniden boyutlandırırken detay kaybını ve artefaktları en aza indirmek istediğinizde kullanılır, özellikle görüntüyü küçültürken. Çok pürüzsüz ve keskin sonuçlar üretir, bu yüzden görüntü kalitesinin önemli olduğu uygulamalarda idealdir.
    """
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
    roi=img_to_array(roi)
    proposals.append(roi)
    boxes.append((x,y,w,h))
proposals=np.array(proposals,dtype="float64")
boxes=np.array(boxes,dtype="int32")

#####################################################

# olasılık değerlerini bulalım
pickel_in=open("../../../EvrisimselSinirAglari/RakamSiniflandirmaProjesi/output/model_trained_new.p","rb")
model=pickle.load(pickel_in)
proba=model.predict(proposals) # olasılık değerlerim

