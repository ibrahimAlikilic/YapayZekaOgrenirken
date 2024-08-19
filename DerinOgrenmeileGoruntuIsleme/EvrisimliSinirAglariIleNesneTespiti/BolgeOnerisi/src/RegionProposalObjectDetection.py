from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import cv2

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MaksimumOlmayanBastirma/src')))
from nunMaxSupression import non_maxi_suppression

#####################################################

def SelectiveSearch(image):
    print("Selective Search aşaması başladı ...")
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    rects=ss.process()
    print("Selective Search aşaması bitti.")
    return rects[:1000]

#####################################################

# Model
model=ResNet50(weights="imagenet")
image=cv2.imread("../input/animals(1).jpg")
image=cv2.resize(image,dsize=(400,400))
(H,W)=image.shape[:2]

#####################################################

# SelectiveSearch Yöntemini çağıralım
rects=SelectiveSearch(image)

# oluşan tüm rectangle lerin içerisinde dalaşarak region of interest (ilgilia alanlar) lerimizi bulacağız
proposals=[]
boxes=[]
for (x,y,w,h) in rects:
    if w/float(W)<0.1 or h/float(H) <0.1: # Bu hesaplamalar, dikdörtgen bölgenin genişliğinin ve yüksekliğinin, orijinal görüntü boyutlarına oranını bulur.
        continue
    roi=image[y:y+h,x:x+w]
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
    roi=cv2.resize(roi,(224,224))
    roi=img_to_array(roi)
    roi=preprocess_input(roi)
    proposals.append(roi)
    boxes.append((x,y,w,h))
    
proposals=np.array(proposals)

# predict (tahmin)
print("Prediction aşaması başladı ...")
preds=model.predict(proposals)
preds=imagenet_utils.decode_predictions(preds,top=1)
print("Prediction aşaması bitti.")

labels={}
min_conf=0.8
print("Sınıflandırma aşaması başlıyor ...")
# Labels oluşum döngüsü
for (i,p) in enumerate(preds):
    (_,label,prob)=p[0]
    if prob>=min_conf:
        (x,y,w,h)=boxes[i]
        box=x,y,x+w,y+h
        L=labels.get(label,[])
        L.append((box,prob))
        labels[label]=L
print("Sınıflandırma aşaması bitti.")
# sınıflandırma sonucu bakalım ne bulmuş
print("****************")
print(f"sınıflandırma sonucunda karar verilen canlı : {labels.keys()}")
print("****************")

clone=image.copy()
    
for label in labels.keys():
    for (box,prob) in labels[label]:
        boxes=np.array([p[0]for p in labels[label]])
        proba=np.array([p[1]for p in labels[label]])
        boxes=non_maxi_suppression(boxes,proba)
        for (startX,startY,endX,endY) in boxes:
            cv2.rectangle(clone,(startX,startY),(endX,endY),(255,0,0),2)
            y=startY-10 if startY-10>10 else startY+10
            cv2.putText(clone,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
        cv2.imshow("After",clone)
cv2.waitKey(0)
cv2.destroyAllWindows()
output_path = os.path.join("..", "output","animals_islem_gormus.png")
cv2.imwrite(output_path, clone)