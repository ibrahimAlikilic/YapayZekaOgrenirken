import cv2
import numpy as np

def non_maxi_suppression(boxes,probs=None,overlapThresh=0.3):
    if len(boxes)==0: # tedbir
        return []
    if boxes.dtype.kind=="i": # date type == intege # 2. tedbir
        boxes=boxes.astype("float")
    
    #####################################################
    
    # koşeleri öğrenelim
    
    # x1 ve y1 resmimin initializ (başlatmak) noktaları
    x1=boxes[:,0]
    y1=boxes[:,1]
    #x2 , y2 bitiş noktası
    x2=boxes[:,2]
    y2=boxes[:,3]
    
    #####################################################
    
    # Alanı öğrenelim
    area=(x2-x1+1)*(y2-y1+1) # terim sayısı gibi düşün kaybettiğimiz 1 pixeli geri ekliyoruz
    
    
    #####################################################
    
    idxs=y2 # default değeri y2 olsun dedi
    # olasılık değerleri
    if probs is not None:
        idxs=probs
    # küçükten büyüğe sıralayalım
    idxs=np.argsort(idxs)
    
    #####################################################
    pick=[] # seçmiş olduğumuz kutularu yazacağız
    while len(idxs)>0:
        last=len(idxs)-1
        i=idxs[last]
        pick.append(i)
        
        # en büyük ve en küçük x ve y değerlerini bulacağız
        xx1=np.maximum(x1[i], x1[idxs[:last]])
        yy1=np.maximum(y1[i], y1[idxs[:last]])
        xx2=np.minimum(x2[i], x2[idxs[:last]])
        yy2=np.minimum(y2[i], y2[idxs[:last]])
        
        # w,h bulalım
        w=np.maximum(0,xx2-xx1+1)
        h=np.maximum(0,yy2-yy1+1)
        
        # overlap bulalım ( Aciklama resminde IoU olarak belirtilen değer .)
        overlap=(w*h)/area[idxs[:last]]
        
        #################################################
        # Threshol umun altında olan index lerimi sileceğim
        idxs=np.delete(idxs,np.concatenate(([last],np.where(overlap>overlapThresh)[0]))) 
    return boxes[pick].astype("int")
"""
Bu fonksiyon, verilen kutular (bounding boxes) üzerinde non-maximum suppression (NMS) işlemi yapar.
NMS, genellikle nesne algılama sistemlerinde, aynı nesneye ait olabilecek birden fazla kutuyu tek bir kutu ile temsil etmek için kullanılır.
Fonksiyonun genel işleyişi, üst üste binen kutular arasından olasılık değeri en yüksek olanı seçip, diğerlerini kaldırmaktır.

Ayrıntılı Açıklama

    Giriş Parametreleri:
        boxes: Üzerinde non-maximum suppression işlemi yapılacak olan kutuların listesidir. Her kutu dört koordinat içerir: [x1, y1, x2, y2], yani sol üst köşe (x1, y1) ve sağ alt köşe (x2, y2).
        probs: Kutularla ilişkili olasılık değerleridir. Bu, bazı durumlarda modelin kutunun içeriğinde bir nesne olduğunu ne kadar güvenli bulduğunu temsil eder.
        overlapThresh: İki kutu arasındaki Intersection over Union (IoU) oranının bu değerin üzerinde olması durumunda, bir kutunun kaldırılacağı eşiği belirler.

    Kutuların Hazırlanması:
        Eğer kutular tamsayı olarak verilmişse, bunlar kayan nokta değerine dönüştürülür. Bu, daha doğru hesaplamalar yapılmasını sağlar.

    Koordinatların Çıkarılması:
        Kutuların başlangıç (x1, y1) ve bitiş (x2, y2) koordinatları, kutular dizisinden çıkarılır.

    Alan Hesaplaması:
        Her kutunun alanı hesaplanır. Bu, daha sonra overlap oranını hesaplamak için kullanılacaktır.

    Olasılık Değerlerinin Sıralanması:
        Eğer olasılık değerleri verilmişse, kutular bu değerlere göre küçükten büyüğe sıralanır.

    NMS Döngüsü:
        pick listesi, nihai olarak seçilen kutuları tutar.
        Döngüde, en yüksek olasılığa sahip kutu seçilir ve bu kutu listenin sonuna eklenir.
        Daha sonra, diğer kutularla bu kutunun ne kadar örtüştüğü hesaplanır. IoU değeri belirli bir eşikten yüksekse, o kutu silinir.
        Döngü, tüm kutular kontrol edilene kadar devam eder.

    Sonuç:
        Seçilen kutular pick listesine eklenir ve fonksiyon bu kutuları döndürür.

Bu fonksiyon, özellikle bir nesne algılama modelinin çıkışındaki fazla kutuları elemek için kullanılır, böylece her nesne için yalnızca bir kutu bırakılır.

"""