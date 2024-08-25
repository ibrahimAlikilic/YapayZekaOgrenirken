import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../input')))
from yolo_model import YOLO 

#########################################################

yolo=YOLO(0.6,0.5)

#########################################################

# Etiketleri çekelim
file="../input/data/data/coco_classes.txt"
with open(file) as f:
    class_names=f.readlines() 
# göremediğimiz boşluklar var onları sileceğiz (benim kaydettiğimde yok ama hocanınkinde \n ler var)
all_classes=[c.strip() for c in class_names]

############################

# Ara Kayit
# class_names ve all_classes değişkenlerini ../output dizinine kaydetmek için
output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))

# output dizini yoksa oluştur
os.makedirs(output_directory, exist_ok=True)

# Dosyaların yollarını belirleyelim
class_names_path = os.path.join(output_directory, 'class_names.txt')
all_classes_path = os.path.join(output_directory, 'all_classes.txt')

# Eğer class_names.txt dosyası yoksa, kaydet
if not os.path.exists(class_names_path):
    with open(class_names_path, 'w') as f:
        f.writelines(class_names)
    print(f"class_names {class_names_path} dizinine kaydedildi")
else:
    print(f"class_names.txt zaten mevcut, işlem yapılmadı.")

# Eğer all_classes.txt dosyası yoksa, kaydet
if not os.path.exists(all_classes_path):
    with open(all_classes_path, 'w') as f:
        for item in all_classes:
            f.write("%s\n" % item)
    print(f"all_classes {all_classes_path} dizinine kaydedildi")
else:
    print(f"all_classes.txt zaten mevcut, işlem yapılmadı.")

#########################################################

# YOLO içim Resim hazırlık

img_path="../input/images/dog_cat.jpg"
image=cv2.imread(img_path)
pimage=cv2.resize(image,(416,416))
pimage=np.array(pimage,dtype="float32")
pimage/=255.0
pimage=np.expand_dims(pimage,axis=0)
"""
Kodun bu satırı, pimage adlı bir NumPy dizisinin boyutunu artırır. Bu işlem, genellikle bir görüntü işleme veya makine öğrenimi bağlamında, modelin girdi olarak beklediği veri şekline uyacak şekilde yapılır.
Detaylı Açıklama:

    np.expand_dims(pimage, axis=0):
        pimage: İşlem yapmak istediğiniz NumPy dizisi (muhtemelen bir görüntü).
        axis=0: Yeni bir boyutun ekleneceği eksen. Bu durumda, ilk eksene (0. eksen) yeni bir boyut ekleniyor.

Ne İşe Yarar?

Örneğin, pimage 3 boyutlu bir görüntü dizisi olsun (örneğin, (height, width, channels) şeklinde). np.expand_dims kullanarak bu diziye ekstra bir boyut eklediğinizde, dizi (1, height, width, channels) boyutuna sahip olur.

Bu işlem, modelin birden fazla görüntü (örneğin, bir minibatch) üzerinde çalışabilmesi için yapılır. Tek bir görüntüyü işlemeye çalışırken bile, model genellikle bir batch boyutu (örneğin, batch_size, height, width, channels) bekler. Bu durumda, batch_size 1 olur ve bu nedenle tek bir görüntüyü işlemek için boyutun genişletilmesi gereklidir.

Özet:

Bu kod satırı, pimage dizisine bir boyut ekleyerek onu modelin beklediği giriş şekline uygun hale getirir. Bu, tek bir görüntü yerine bir batch olarak işlenebilmesi için yapılır.
"""

#########################################################

# Predict aşaması
boxes,classes,scores=yolo.predict(pimage,image.shape)
print(f"boxes : {boxes}")
print(f"classes : {classes}")
print(f"scores : {scores}")

# sınıflandırmayı yaptık şimdi doğru tespit edebilmiş miyiz diye görselleştirme yapacağız
for box,cl,sco in zip(boxes,classes,scores):
    x,y,w,h=box
    # hatırlatma : np.floor olunca ondalıklı sayı taban değere yuvarlanır.
    top=max(0,np.floor(x+0.5).astype(int))
    left=max(0,np.floor(y+0.5).astype(int))
    right=max(0,np.floor(x+w+0.5).astype(int))
    bottom=max(0,np.floor(y+h+0.5).astype(int))
    cv2.rectangle(image,(top,left),(right,bottom),(255,0,0),2)
    cv2.putText(image,"{} {}".format(all_classes[cl],sco),(top,left-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1,cv2.LINE_AA)

cv2.imshow("YOLO",image)
cv2.waitKey(0)
cv2.destroyAllWindows()