import cv2
import pickle
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array
import os


image=cv2.imread("../input/mnist(1).png") 

#####################################################

print("Selective Search aşaması başladı ...")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()
rects=ss.process()
print("Selective Search aşaması bitti.")

#####################################################

# Sınıflandırma

proposals=[]
boxes=[]
output=image.copy()
for (x,y,w,h) in rects[:100]:
    color=[random.randint(0,255)for j in range(0,3)]
    roi=image[y:y+h,x:x+w]
    roi=cv2.resize(roi,dsize=(32,32),interpolation=cv2.INTER_LANCZOS4)
    """
    cv2.INTER_LANCZOS4 nedir?

    Lanczos Enterpolasyonu (Lanczos-4): Bu yöntem, bir sinc fonksiyonuna dayanan yüksek kaliteli bir enterpolasyon algoritması kullanır (Lanczos çekirdeği). INTER_LANCZOS4'deki 4, çekirdekte kullanılan lob sayısını ifade eder ve bu, enterpolasyonun pürüzsüzlüğünü ve doğruluğunu etkiler.

cv2.INTER_LANCZOS4 ne zaman kullanılır?

    Yüksek kaliteli yeniden boyutlandırma: INTER_LANCZOS4, bir görüntüyü yeniden boyutlandırırken detay kaybını ve artefaktları en aza indirmek istediğinizde kullanılır, özellikle görüntüyü küçültürken. Çok pürüzsüz ve keskin sonuçlar üretir, bu yüzden görüntü kalitesinin önemli olduğu uygulamalarda idealdir.
    """
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    roi=img_to_array(roi)
    proposals.append(roi)
    boxes.append((x,y,x+w,y+h))
proposals=np.array(proposals,dtype="float64")
boxes=np.array(boxes,dtype="int32")


# olasılık değerlerini bulalım
pickel_in=open("../../../EvrisimselSinirAglari/RakamSiniflandirmaProjesi/output/model_trained_new.p","rb")
model=pickle.load(pickel_in)
proba=model.predict(proposals) # olasılık değerlerim

number_list=[]
idx=[]
for i in range(len(proba)):
    max_proba=np.max(proba[i,:])
    if max_proba>0.95:
        idx.append(i)
        number_list.append(np.argmax(proba[i])) # max değerin index ini ekledik
        
#####################################################

# Görselleştirme
for i in range(len(number_list)):
    j=idx[i]
    cv2.rectangle(image,(boxes[j,0],boxes[j,1]),(boxes[j,2],boxes[j,3]),(0,0,255),2)
    cv2.putText(image,str(np.argmax(proba[j])),(boxes[j,0]+5,boxes[j,1]+5),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),1)
image=cv2.resize(image,(650,650)) # çıktı için orijinal boyutu beğenmedim
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#####################################################

# Kayit
output_path = os.path.join("..", "output", "siniflandirilmisRakamlar.png")
cv2.imwrite(output_path, image)
print("****************")
print(f"siniflandirilmisRakamlar için görüntü '{output_path}' dosyasına kaydedildi.")
print("****************")