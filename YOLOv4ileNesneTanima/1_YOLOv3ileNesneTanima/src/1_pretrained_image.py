import cv2
import numpy as np
import random

##########################################

# Resim
img=cv2.imread("../input/images/people.jpg") # Hatırlatma : Resimleri BGR okur
img_width=640
img_height=427
img=cv2.resize(img,(img_width,img_height))
print(f"img shape : {img.shape}") # hatırlatma : height , weidth olarak verir

##########################################

# resmimizi blob formata çevirmemiz lazım ( resmin 4 boyutlu tensorlere çevrilmiş hali )
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

# Değişkenleri (variable) oluşturalım

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
print(f"type(colors) : {type(colors)} , colors.dtype : {colors.dtype}") # ileride tr sorunu yaşamamak için kendimi teğit ediyorum # float dedi bana int lazım
colors = colors.astype(int)
print(f"type(colors) : {type(colors)} , colors.dtype : {colors.dtype}") # ileride tr sorunu yaşamamak için kendimi teğit ediyorum # float dedi bana int lazım
