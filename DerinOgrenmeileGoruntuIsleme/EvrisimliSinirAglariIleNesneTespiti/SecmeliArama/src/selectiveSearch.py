import cv2
import random
import os

#####################################################

image = cv2.imread("../input/pyramid.jpg")
image = cv2.resize(image, dsize=(600, 600))

#####################################################

# ss nesnesini ilklendir
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
"""
1. ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

Bu satır, Selective Search Segmentation işlemini başlatmak için gerekli olan bir nesneyi oluşturur.

    cv2.ximgproc.segmentation: 
    OpenCV'nin ekstra modüllerini içeren ximgproc (Extended Image Processing) modülünün altındaki 
    segmentation (segmentasyon) sınıfıdır. Segmentasyon, bir görüntüyü daha küçük anlamlı parçalara ayırma işlemidir.

    createSelectiveSearchSegmentation(): 
    Bu yöntem, SelectiveSearchSegmentation nesnesini oluşturur.
    Bu nesne, daha sonra segmentasyon işlemi için kullanılacaktır. Selective Search algoritması, 
    bir görüntüdeki potansiyel nesne bölgelerini bulmak için kullanılan bir algoritmadır.
    Bu algoritma, görüntüyü farklı ölçekteki segmentlere ayırarak olası nesne bölgelerini tanımlar.
    
        Amaç: 
        Bu komut, Selective Search Segmentasyon sürecini başlatmak için gerekli nesneyi oluşturur ve
        bu nesne üzerinde daha sonra başka işlemler yapılabilir.

2. ss.setBaseImage(image)

Bu satır, segmentasyon işlemi yapılacak temel görüntüyü tanımlar.

    ss: Önceki satırda oluşturulan SelectiveSearchSegmentation nesnesidir.

    setBaseImage(image): Bu yöntem, üzerinde segmentasyon işlemi yapılacak olan görüntüyü (base image) ayarlar.

        image: Bu, işlem yapmak istediğiniz görüntüyü temsil eder. 
        Örneğin, bir RGB görüntü, bir numpy array'i olarak bu değişkene atanabilir.

        Amaç: Bu komut, SelectiveSearchSegmentation nesnesine hangi görüntü üzerinde çalışması gerektiğini bildirir. 
        Segmentasyon işlemleri bu görüntü üzerinde gerçekleştirilecektir.

Özet:

Bu iki satır, Selective Search algoritmasını kullanarak bir görüntüdeki nesne bölgelerini tespit etmek için 
gerekli nesneyi oluşturur ve ardından hangi görüntü üzerinde çalışacağını belirtir. 
Bu işlemin amacı, görüntüdeki farklı bölgeleri anlamlı segmentlere ayırmak ve 
bu bölgeleri daha sonra başka işlemler için (örneğin, nesne tespiti) kullanmaktır.
"""

ss.switchToSelectiveSearchQuality()

"""
ss.switchToSelectiveSearchQuality() yöntemi, Selective Search Segmentasyon algoritmasının "kalite moduna" geçmesini sağlar. Bu mod, daha yüksek bir doğrulukla nesne bölgelerini bulmayı amaçlar, ancak işlem süresi daha uzundur.
Detaylı Açıklama:

    ss: Bu, önceki satırlarda oluşturulan SelectiveSearchSegmentation nesnesidir.

    switchToSelectiveSearchQuality(): Bu yöntem, algoritmanın daha hassas ve detaylı bir şekilde çalışmasını sağlar. "Kalite modu", algoritmanın daha fazla olası nesne bölgesini tespit etmesini ve bu bölgelerin daha iyi segmentlere ayrılmasını hedefler. Bununla birlikte, bu mod, daha fazla işlem gücü gerektirir ve daha yavaş çalışır.

Amaç:

    Kalite Modu: Kalite moduna geçmek, segmentasyon işleminin doğruluğunu artırır. Bu, özellikle nesne tespitinin doğru bir şekilde yapılmasının kritik olduğu durumlarda kullanılır. Ancak, bu modun daha fazla zaman alacağı unutulmamalıdır.

Özet:

switchToSelectiveSearchQuality(), segmentasyon işleminin doğruluğunu artırmak için algoritmayı daha hassas bir moda geçirir. Bu, işlem süresini uzatabilir, ancak daha iyi sonuçlar elde etmenizi sağlar.
"""
print("basla...")
rects=ss.process()
print("bitti")
"""
rects = ss.process() satırı, Selective Search Segmentasyon algoritmasının temel görüntü üzerinde çalıştırılmasını sağlar ve olası nesne bölgelerini dikdörtgenler (rectangles) olarak döndürür.
Detaylı Açıklama:

    rects: Bu değişken, ss.process() fonksiyonunun çıktısını saklar. Bu çıktı, görüntüdeki olası nesne bölgelerini temsil eden dikdörtgenlerin (rectangles) bir listesidir. Her dikdörtgen, bir tuple (demet) şeklinde, dört değer içerir: (x, y, w, h).
        x ve y: Dikdörtgenin sol üst köşesinin koordinatları.
        w (width): Dikdörtgenin genişliği.
        h (height): Dikdörtgenin yüksekliği.

    ss.process(): Bu yöntem, daha önce ayarlanmış temel görüntü (setBaseImage(image)) üzerinde Selective Search Segmentasyon algoritmasını çalıştırır. Algoritma, olası nesne bölgelerini bulur ve bu bölgeleri, tespit edilen dikdörtgenlerin koordinatları olarak döndürür.
        Amaç: Görüntüdeki olası nesne bölgelerini tespit etmek ve bu bölgeleri dikdörtgenler halinde döndürmek. Bu dikdörtgenler daha sonra nesne algılama veya sınıflandırma gibi diğer işlemler için kullanılabilir.

Özet:

rects = ss.process() komutu, Selective Search Segmentasyon algoritmasını çalıştırarak, görüntüdeki potansiyel nesne bölgelerini dikdörtgenler şeklinde tespit eder ve bu dikdörtgenleri bir liste olarak döndürür. Bu liste, görüntüdeki her bir potansiyel nesnenin konumunu ve boyutunu içerir.
"""

#####################################################

# sınıflandırılmak isteyen kutucuklarımdan ilk 50 tanesini göreyim
output=image.copy()
for (x,y,w,h) in rects[:50]:
    color=[random.randint(0,255) for j in range(0,3)]
    cv2.rectangle(output,(x,y),(x+w,y+h),color,2)

cv2.imshow("output",output)
cv2.waitKey(0)
cv2.destroyAllWindows()
output_path = os.path.join("..", "output", "Cikti.png")
cv2.imwrite(output_path, output) 
print("Cikti kaydedildi.")