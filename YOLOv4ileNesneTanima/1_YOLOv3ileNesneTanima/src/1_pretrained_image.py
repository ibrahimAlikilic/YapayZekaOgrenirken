#%% Kutuphaneler
import cv2
import numpy as np
import random

##########################################
#%% Resim
img=cv2.imread("../input/images/neighbourhood.jpg") # Hatırlatma : Resimleri BGR okur
img_width=640
img_height=427
img=cv2.resize(img,(img_width,img_height))
print(f"img shape : {img.shape}") # hatırlatma : height , weidth olarak verir

##########################################
#%% resmimizi blob formata çevirmemiz lazım ( resmin 4 boyutlu tensorlere çevrilmiş hali )

"""
Yapay zekada bir resmi "blob" formata çevirmek, bir resmi veya görüntüyü, daha sonra işlenmek veya analiz edilmek üzere bir ikili veri akışına (binary large object - BLOB) 
dönüştürmek anlamına gelir. Bu format, özellikle bilgisayar görüşü ve makine öğrenimi gibi alanlarda yaygın olarak kullanılır.
Neden Blob Formatı Kullanılır?

    Veri Depolama: Blob formatı, resim gibi büyük veri dosyalarını veri tabanlarında veya bellekte saklamak için uygun bir formattır.

    Veri Aktarımı: Blob formatı, bir resmi veya görüntüyü ağ üzerinden veya farklı programlama ortamları arasında kolayca aktarmayı sağlar.

    Veri İşleme: Blob formatındaki bir görüntü, makine öğrenimi modelleri, görüntü işleme algoritmaları veya yapay zeka uygulamaları tarafından daha kolay işlenebilir.
blob, bir yapay zeka modeline girdi olarak verilebilecek şekilde hazırlanmış bir veri formatıdır.
"""
"""
Tensör, matematikte ve bilgisayar bilimlerinde çok boyutlu bir veri yapısını ifade eder.
Tensörler, genellikle makine öğrenimi, yapay zeka ve özellikle derin öğrenme modellerinde kullanılır.
Tensörler, veriyi organize etmek ve işlemek için kullanılır ve temel olarak, farklı boyutlarda (skalarlardan matrislere kadar) veri yapılarını temsil edebilir.

Tensörlerin Boyutları

    Skalar: Sıfır boyutlu bir tensördür. Tek bir sayıyı ifade eder. Örneğin, x = 5.

    Vektör: Bir boyutlu bir tensördür. Bir dizi sayıyı içerir. Örneğin, [1, 2, 3].

    Matris: İki boyutlu bir tensördür. Satır ve sütunlar halinde düzenlenmiş bir sayı tablosudur. Örneğin, bir 2x3 matrisi
    Üç Boyutlu Tensör: Üç boyutlu bir tensördür. Birden fazla matrisin birleşimidir. Örneğin, bir dizi 2x3 matris
        Dört Boyutlu Tensör: Dört boyutlu bir tensör, genellikle bir görüntü kümesiyle çalışırken kullanılır.
        Örneğin, bir video veya birden fazla görüntü kümesi olarak düşünülebilir. Bu tensör genellikle Batch size x Channels x Height x Width şeklinde organize edilir.
        Her bir bileşen şu anlama gelir:
        
        Batch size: Aynı anda işlenen görüntü sayısı.
        Channels: Renk kanalları (örneğin, RGB için 3 kanal).
        Height ve Width: Görüntünün yüksekliği ve genişliği.

Tensörlerin Kullanımı

Tensörler, özellikle derin öğrenme modellerinde çok önemlidir çünkü modeller genellikle bu tensörleri giriş olarak alır, üzerinde çeşitli işlemler yapar ve çıktıyı yine tensör formatında verir. Örneğin, bir görüntü tanıma modeline bir resim verildiğinde, bu resim önce bir tensöre dönüştürülür, model tarafından işlenir ve sonucunda bir sınıflandırma kararı verir.
"""
img_blob=cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)
                              # resim , yolonun yazarlarının kabulü = 1/255 , indirmiş olduğumuz modele göre remimizin boyutunu resize2liyoruz , BGT2RGB , crop=kırpma
print(f"img_blob shape : {img_blob.shape}") # hatırlatma : height , weidth olarak verir

##########################################
#%% Değişkenleri (variable) oluşturalım

# labels.txt dosyasını oku
path="../input/"
with open(path+'labels.txt', 'r') as file:
    data = file.read()

# Gereksiz karakterleri temizle ve listeye çevir
labels = data.replace('"', '').split(',')

# labels listesini kontrol et
print("****************************")
print(f"labels : {labels}")


colors = np.random.uniform(0, 255, size=(len(labels), 3))
print("****************************")
print(f"type(colors) : {type(colors)} , colors.dtype : {colors.dtype}") # ileride tür sorunu yaşamamak için kendimi teğit ediyorum # float dedi bana int lazım
colors = colors.astype(int)
print(f"type(colors) : {type(colors)} , colors.dtype : {colors.dtype}") # ileride tür sorunu yaşamamak için kendimi teğit ediyorum # float dedi bana int lazım

#####################
# Model variable
model=cv2.dnn.readNetFromDarknet("../../../../Masaüstü/YOLOlar/YOLOv3/pretrained_model/yolov3.cfg","../../../../Masaüstü/YOLOlar/YOLOv3/pretrained_model/yolov3.weights")
"""
Bu kod satırı, OpenCV'nin DNN (Deep Neural Network) modülünü kullanarak YOLOv3 modelini yüklemektedir. Kodun detaylarına bakalım:
cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    cv2.dnn.readNetFromDarknet: Bu fonksiyon, YOLO (You Only Look Once) gibi Darknet tabanlı bir derin öğrenme modelini OpenCV'nin DNN modülüne yükler.

    configPath (birinci parametre): Bu, YOLOv3 modelinin yapılandırma (config) dosyasının yolunu belirten parametredir.
    Bu dosya (yolov3.cfg), modelin katman yapısını, parametrelerini ve diğer gerekli yapılandırmaları içerir. Senin durumunda bu yol:


    weightsPath (ikinci parametre): Bu, önceden eğitilmiş modelin ağırlıklarının bulunduğu dosyanın yoludur.
    Bu dosya (yolov3.weights), modelin ağırlıklarını yani öğrenilmiş parametrelerini içerir. Senin durumunda bu yol:

Ne Yapar?

Bu kod çalıştırıldığında, YOLOv3 modelini OpenCV'ye yükler, böylece bu modeli görüntü işlemede, özellikle nesne tespiti için kullanabilirsin.
Model, cfg dosyasındaki yapılandırmaya ve weights dosyasındaki ağırlıklara göre nesne tespiti yapacaktır.

Bu kod satırını çalıştırdıktan sonra, model üzerinde görüntüleri işleyecek ve belirlenen nesneleri tanıyacak şekilde çalışabilirsin.
"""

layers = model.getLayerNames() # layer = katman
"""
getLayerNames(): Bu yöntem, modelin tüm katmanlarının isimlerini içeren bir liste döndürür.
Katmanlar, bir derin öğrenme modelinde, verilerin işlenme aşamalarını ifade eder.
YOLOv3 gibi modellerde, birçok farklı katman bulunur ve her katman, belirli bir görevi yerine getirir (örneğin, evrişim, aktivasyon, birleştirme, vb.).
"""

# layers tğm katmanları içeriyor benim istediğim şey detection (tespit) ( yani çıktı ) katmanları
# bunun için özel bir fonksiyon var ama onda şöyle bir durum var bze kaçıncı sırada olduğunu döndürüyor ama biz index olarak kullanıyoruz o yüzden -1 diyeceğiz
print("****************************")
print(f"model.getUnconnectedOutLayers() :  {model.getUnconnectedOutLayers()}")
output_layert=[layers[layer-1] for layer in model.getUnconnectedOutLayers()]
# Eğitim videosunda layer[0]-1 diyordu ama hata aldım ve benim mantığıma da uymamıştı yanlış olduğunu düşünüyordum ki hata aldım o yüzden bu şekilde düzenledim

model.setInput(img_blob)
# setInput: Bu yöntem, modele bir giriş (input) verisi sağlar. Yani, modelin işlem yapması için gerekli olan veriyi, bu fonksiyon aracılığıyla modele iletmiş oluyorsun.
detection_layers=model.forward(output_layert)
# forward() fonksiyonu, modelin ileri besleme işlemini gerçekleştirir. Bu, modelin giriş verisini (görüntüyü) işleyip çıktıları (nesne tespitlerini) üretmesi anlamına gelir.
# Bu fonksiyon, modelin belirtilen çıkış katmanlarının çıktısını döndürür.

##########################################
#%% İnceleme
print("****************************")
print(detection_layers) # gördüğümüz üzere belli başlı kendi içerisinde manası olan matrislere sahipler

# bu matrisleri gezelim
# çok boyutlu matris olaraka düşün o yüzden 2 for
print("****************************")
for detection_layer in detection_layers:
    for object_detection in detection_layer:
        # güven skorü ile ilgilenelim
        score=object_detection[5:] # ilk 5 değer box ile ilgili
        # score içerisindeki en büyük değer benim predicted id olacak = tahmin edilen nesnemin index i
        predicted_id=np.argmax(score)
        confidence=score[predicted_id] # confidence = güven skoru
        
        # box çizim aşamasına geldik
        if confidence > 0.3:
            label=labels[predicted_id] # labeli ne onu alalım
            bounding_box=object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
            # object_detection[0:4] ün döndürdüğü değerler normalize edilmiş durumdadır bu sebepten dolayı " * np.array([img_width,img_height,img_width,img_height])" yaıyoruz
            (box_center_x,box_center_y,box_width,box_height)=bounding_box.astype("int")
            start_x=int(box_center_x-(box_width/2))
            start_y=int(box_center_y-(box_height/2))
            end_x = start_x + box_width
            end_y = start_y + box_height
            
            box_color=colors[predicted_id]
            box_color=[int(each) for each in box_color]# liste şeklinde olması gerekemkte # aslında yukarıda int olarak oluşturmuştum ama burada int olmayınca hata aldım
            cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,1)
            
            # label imde confidence değerimi de görmek istiyorum
            label="{}: {:.2f}%".format(label,confidence*100) # Hatırlatma : ".2f" . dan sonra 2 basamak olsun manasında
            print(f"prediction object {label}") # uçbirimde de görmek istedim
            cv2.putText(img,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
# Büyük görmek istiyorum
img=cv2.resize(img,(1280,960))
cv2.imshow("Tespit (Detection)",img)
cv2.waitKey(0)
cv2.destroyAllWindows()