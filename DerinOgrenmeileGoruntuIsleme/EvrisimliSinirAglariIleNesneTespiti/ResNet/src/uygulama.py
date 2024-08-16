# from tensorflow.keras.applications.resnet50 import preprocess_input # aşağıda tanımladık gerek kalmadı
# from tensorflow.keras.applications import ResNet50 gpt hatalı yazım dedi
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import cv2
import pandas as pd
import os

#####################################################

# Başka klasöreki fonksiyonu import etme.

import sys
import os
# İmport etedeğim fonksiyonunun py dosyasının olduğu dizine git.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../PiramitGosterimi/src')))
# import et
from imagePyramid import imagePyramid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MaksimumOlmayanBastirma/src')))
from nunMaxSupression import non_maxi_suppression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../KayanPencere/src')))
from slidingWindow import slidinWindow

#####################################################

# initialize paramtrelerini belirleme :
WIDTH=600
HEIGHT=600
PYR_SCALE=1.5 # Resim piramidimizin scale değeri
WIN_STEP=16 # slidingWindow un step size si
ROI_SIZE=(200,150)
INPUT_SIZE=(224,224) # ResNet CNN imizin input size'ı # nORAL nETWORK Ü BU BUYOTA GÖRE EĞİTTİKLERİ İÇİN BU BOYUTU VERDİK

#####################################################

# ResNet Yükleme
print("****************")
print("Resnet yüklenmeye başlıyor...")
# ResNet internetten ağırlıklarını indirecek
model=ResNet50(weights="imagenet",include_top=True) # rennet in sabit değerlerini parametreye veriyoruz
print("Resnet yüklendi .")
print("****************")

"""
Bu kod satırı, TensorFlow ve Keras kütüphanelerini kullanarak bir ResNet50 modelini oluşturur. Şimdi adım adım bu kod satırını açıklayalım:

    ResNet50 fonksiyonu:
        ResNet50, Keras uygulamaları (applications) modülünde tanımlı bir fonksiyondur ve ResNet-50 mimarisine sahip bir derin öğrenme modelini oluşturur.
        ResNet-50, 50 katmanlı bir derin sinir ağıdır ve görüntü sınıflandırma problemlerinde yaygın olarak kullanılır.
        Bu model, derin öğrenme topluluğunda oldukça popülerdir çünkü derinliği artırarak daha karmaşık özellikleri öğrenebilirken,
        "residual connections" (kalıntı bağlantıları) kullanarak vanishing gradient (kaybolan gradyan) sorununu önler.

    weights="imagenet" argümanı:
        weights parametresi, modelin önceden eğitilmiş ağırlıklarını belirtir.
        "imagenet" değeri, modelin ImageNet veri seti üzerinde eğitilmiş olduğunu belirtir.
        ImageNet, milyonlarca görüntü ve 1000 farklı sınıfa sahip büyük bir veri setidir.
        Bu, modelin belirli bir görev için yeniden eğitilmeden önce genel özellikleri öğrenmiş olduğu anlamına gelir.
        
        Eğer "imagenet" yerine None yazılsaydı, model rastgele ağırlıklarla başlatılacaktı.

    include_top=True argümanı:
        include_top, modelin son katmanlarını (genellikle sınıflandırma katmanları olarak adlandırılır) içerip içermeyeceğini belirtir.
        True olarak ayarlandığında, ResNet-50'nin orijinal mimarisine uygun olarak tam model yüklenir,
        yani ImageNet'teki 1000 sınıfa uygun olan tamamen bağlı (fully connected) katmanlar dahil edilir.
        
        Eğer include_top=False olsaydı, modelin üst kısmı (sınıflandırma katmanları) dahil edilmezdi,
        bu da genellikle transfer öğrenme yaparken kullanılır. Böylece, yeni bir veri setine uygun özel sınıflandırma katmanları eklenebilir.

Özetle: Bu kod satırı, ImageNet veri seti üzerinde önceden eğitilmiş ResNet-50 modelini yükler ve modelin tam yapısını,
yani sınıflandırma katmanları dahil olacak şekilde oluşturur. Bu model, 1000 sınıflı bir görüntü sınıflandırma problemi için doğrudan kullanılabilir.
"""

#####################################################

# Tespit için reimimizi hazırlayalım
orig=cv2.imread("../input/husky.jpg")
# İlk başta boyutunu slidinWindow,imagePyramid uygun olacak hale getirelim en son modelimize sokarken ResNet boyutuna getireceğiz
orig=cv2.resize(orig,dsize=(WIDTH,HEIGHT))

# H,W parametrelerimize de 600,600 atayalım
(H,W)=orig.shape[:2]

#####################################################

# image pyramid
# burada resimlerin skalasını ( ölçeğini ) değiştiriyorduk
pyramid=imagePyramid(orig,PYR_SCALE,ROI_SIZE)

#####################################################

# slidingWindow : 
# image pyramid her seferinde yeni bir resim ortaya çıkarıyor biz de her resimde slidingWindow yöntemi ile her bir pencerenin içerisindeki resmi sınıflnadıracağız
# anlatmış olduğum işlemin sonucunda bazı sonuçlar çıkacak onları oluşturalım

rois=[]
locs=[]

for image in pyramid:
    # window size ı da değiştirmemiz lazım çnkü elimizdeki resim ile bize döndürülen pencerenin boyutu farkı olursa uyumsuzluk olur
    scale=W/float(image.shape[1])
    for (x,y,roiOrig) in slidinWindow(image,WIN_STEP,ROI_SIZE): # roiOrig = bizim kutucuğumuzun içerisinde bulunan resim
        x=int(x*scale)
        y=int(y*scale)
        w=int(ROI_SIZE[0]*scale)
        h=int(ROI_SIZE[1]*scale)
        
        # roiOrig sınıflandırmada kullanabileceğim hale getirmem gerekiyor
        # yani terimsel olarak : preprocessing 
        roi=cv2.resize(roiOrig,INPUT_SIZE)
        roi=img_to_array(roi)
        roi=preprocess_input(roi) # artk ResNet için hazır
        rois.append(roi)
        locs.append((x,y,x+w,y+h))

# rois lerimi np.array haline getirmem lazım
rois=np.array(rois,dtype="float32")

# Sınıflandırma Aşaması :
print("****************")
print("Sınıflandırma işlemi başlıyor ...")
preds = model.predict(rois)  # Tahminleri döndürür.
print("Sınıflandırma işlemi bitti .")
print("****************")

#####################################################

# exel olarak yadetme
# eğitimde bu yok ama ben inceleme daha rahat yapıldığını düşündüğümden ekledim

# İlk tahminleri kaydetmek için bir liste
# burada pek neyin ne olduğunu anlayamıyoruz ve decode etmemiz lazım diyoruz
initial_results = []

for i, pred in enumerate(preds):
    # pred bir batch'te tek tahmin ise, bunu bir batch'e dönüştürüyoruz
    if pred.ndim == 1:
        pred = np.expand_dims(pred, axis=0)
    
    # İlk sonuçları (decode edilmemiş) kaydet
    initial_results.append({
        "Image_Index": i,
        "Predictions": pred.tolist(),  # Numpy array olduğu için listeye dönüştürülmesi gerekiyor
        "Bounding_Box": locs[i]
    })

# İlk sonuçları DataFrame olarak kaydetme
df_initial = pd.DataFrame(initial_results, columns=["Image_Index", "Predictions", "Bounding_Box"])

# Kaydetme konumunu belirt
output_path_initial = os.path.join("..", "output", "predictions_initial.xlsx")
df_initial.to_excel(output_path_initial, index=False)
print("****************")
print(f"İlk tahminler '{output_path_initial}' dosyasına kaydedildi.")

###########################

# preds'i decode etme
preds = imagenet_utils.decode_predictions(preds, top=1)

# Decode edilmiş tahminlerin işlenmesi ve tabloya dönüştürülmesi
decoded_results = []
for i, decoded_pred in enumerate(preds):
    (imagenetID, label, prob) = decoded_pred[0]
    
    # Tablonun satırı olarak tahmin sonuçlarını kaydet
    decoded_results.append({
        "Image_Index": i,
        "Label": label,
        "Probability": prob,
        "Bounding_Box": locs[i]
    })

# Decode edilmiş sonuçları DataFrame olarak kaydetme
df_decoded = pd.DataFrame(decoded_results, columns=["Image_Index", "Label", "Probability", "Bounding_Box"])

# Kaydetme konumunu belirt
output_path_decoded = os.path.join("..", "output", "predictions_decoded.xlsx")
df_decoded.to_excel(output_path_decoded, index=False)

print(f"Decode edilmiş tahminler '{output_path_decoded}' dosyasına kaydedildi.")
print("****************")

#####################################################

# şimdi olasılıklara göre sınıflandırıcı tanımlayacağız
labels={}
min_conf=0.9

# Labels oluşum döngüsü
for (i,p) in enumerate(preds):
    (_,label,prob)=p[0]
    if prob>=min_conf:
        box=locs[i]
        
        L=labels.get(label,[])
        L.append((box,prob))
        labels[label]=L
# sınıflandırma sonucu bakalım ne bulmuş
print("****************")
print(f"sınıflandırma sonucunda karar verilen canlı : {labels.keys()}") # Eskimo_dog olarak ayıklayabilmiş olduğumuzu gördük.
print("****************")

ilkResimKayit=True
ikinciResimKayit=True
for label in labels.keys():
    clone=orig.copy()
    
    # kutucuklarımızı çizelim
    for (box,prob) in labels[label]:
        (startX,startY,endX,endY)=box
        cv2.rectangle(clone,(startX,startY),(endX,endY),(255,0,0),2)
    
    if ilkResimKayit: 
        # clone'u PNG olarak kaydedelim
        output_path = os.path.join("..", "output", f"{label}_cok_alan_algilanmis.png")
        cv2.imwrite(output_path, clone) # görüldüğü üzeri birden çok şeyi eskimo olarak sınıflandırmışız ve yanlış bizim istediğimiz bu değil
        print("****************")
        print(f"{label} için görüntü '{output_path}' dosyasına kaydedildi.")
        print("****************")
        ilkResimKayit=False
    
    clone=orig.copy()
    # non-maxima
    boxes=np.array([p[0] for p in labels[label]])
    proba=np.array([p[1] for p in labels[label]])
    
    boxes=non_maxi_suppression(boxes,proba)
    
    # kutucuklarımızı yapmakla devam edelim
    for (startX,startY,endX,endY) in boxes:
        cv2.rectangle(clone,(startX,startY),(endX,endY),(255,0,0),2)
        y=startY-10 if startY-10>10 else startY+10
        cv2.putText(clone,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
    
    if ikinciResimKayit:
        # clone'u PNG olarak kaydedelim
        output_path = os.path.join("..", "output", f"{label}_tek_alan_algilanmis.png")
        cv2.imwrite(output_path, clone) # görüldüğü üzeri birden çok şeyi eskimo olarak sınıflandırmışız ve yanlış bizim istediğimiz bu değil
        print("****************")
        print(f"{label} için görüntü '{output_path}' dosyasına kaydedildi.")
        print("****************")
        ikinciResimKayit=False