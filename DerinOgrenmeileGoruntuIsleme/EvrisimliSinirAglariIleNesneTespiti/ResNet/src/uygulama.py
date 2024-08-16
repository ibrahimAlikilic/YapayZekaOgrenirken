# from tensorflow.keras.applications.resnet50 import preprocess_input # aşağıda tanımladık gerek kalmadı
# from tensorflow.keras.applications import ResNet50 gpt hatalı yazım dedi
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import cv2

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
print("Resnet yüklenmeye başlıyor...")
# ResNet internetten ağırlıklarını indirecek
model=ResNet50(weights="imagenet",include_top=True) # rennet in sabit değerlerini parametreye veriyoruz
print("Resnet yüklendi .")
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
orig=cv2.imread("../input/husjy.jpg")
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

roi=[]
locs=[]

