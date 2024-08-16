import cv2
import matplotlib.pyplot as plt

def slidinWindow(image,step,ws):
    for y in range(0,image.shape[0]-ws[1],step):
        for x in range(0,image.shape[1]-ws[0],step):
            yield(x,y,image[y:y+ws[1],x:x+ws[0]])

"""
def slidinWindow(image, step, ws):

    image: Üzerinde işlem yapılacak olan görüntü. Bu, genellikle bir NumPy dizisi olarak temsil edilir. NumPy dizileri, .shape özelliği ile boyut bilgilerini (yükseklik, genişlik, kanal sayısı) tutar.
    step: Kaydırılabilir pencerenin her adımda ne kadar hareket edeceğini belirten piksel sayısıdır. Pencere her adımda sağa ve aşağıya bu kadar piksel kaydırılır.
    ws: Kaydırılabilir pencerenin boyutunu belirten bir tuple. Bu, pencerenin genişlik ve yüksekliğini içerir (örneğin, (width, height)).

Döngülerin Çalışma Prensibi

    for y in range(0, image.shape[0] - ws[1], step):

    range(0, image.shape[0] - ws[1], step):
        image.shape[0]: Görüntünün yüksekliğini ifade eder.
        ws[1]: Pencerenin yüksekliği.
        range fonksiyonu, y değişkeninin değerlerini oluşturur ve 0'dan başlar. image.shape[0] - ws[1] bitiş noktasıdır, yani pencerenin alt kenarının görüntü sınırlarını aşmaması için bir sınır konur.
        step: Her adımda y koordinatı bu miktarda artırılır, yani pencere aşağıya kaydırılır.

        for x in range(0, image.shape[1] - ws[0], step):

    range(0, image.shape[1] - ws[0], step):
        image.shape[1]: Görüntünün genişliğini ifade eder.
        ws[0]: Pencerenin genişliği.
        range fonksiyonu, x değişkeninin değerlerini oluşturur ve 0'dan başlar. image.shape[1] - ws[0] bitiş noktasıdır, yani pencerenin sağ kenarının görüntü sınırlarını aşmaması için bir sınır konur.
        step: Her adımda x koordinatı bu miktarda artırılır, yani pencere sağa kaydırılır.

yield İfadesi

            yield (x, y, image[y:y+ws[1], x:x+ws[0]])

    Bu satır, fonksiyonun en önemli kısmıdır. yield, Python'da bir generator oluşturur, bu da fonksiyonun her çağrıldığında bir sonraki pencereyi döndüreceği anlamına gelir.
    (x, y, image[y:y+ws[1], x:x+ws[0]]):
        x ve y: Pencerenin sol üst köşesinin koordinatları.
        image[y:y+ws[1], x:x+ws[0]]: Görüntüden alınan pencere alanı. y:y+ws[1] aralığı pencerenin dikey olarak kapsadığı pikselleri, x:x+ws[0] aralığı ise yatay olarak kapsadığı pikselleri ifade eder.
        Bu ifade, pencerenin o anki konumunda görüntüden bir alt bölüm alır ve döndürür.

Fonksiyonun Genel Çalışma Prensibi

Bu fonksiyon, verilen bir görüntü üzerinde kaydırılabilir bir pencere tanımlar ve bu pencereyi adım adım sağa ve aşağıya kaydırarak tüm görüntü üzerinde dolaşır. Her adımda pencerenin o anki konumunu (x, y) ve bu konumda yer alan görüntü parçasını geri döndürür. Bu tür bir pencereleme işlemi, nesne algılama, özellik çıkarma veya lokalize edilmiş görüntü işlemleri için kullanışlıdır. Bu sayede görüntü üzerinde belirli boyutlardaki bölgeler analiz edilebilir.

"""

"""
Görmek amacıyla oluşturmuştuk başka koda çağırırken olmaması lazım

img=cv2.imread("../input/husky.jpg")
im=slidinWindow(img,5,(200,150))
for i,image in enumerate(im):
    print(f"{i}. resim oluşturuldu")
    # bir tanesini görelim
    if i == 14125: # i yi arttırdıkça netlik azalmış halini görüyoruz
        print(image[0],image[1])
        plt.imshow(cv2.cvtColor(image[2], cv2.COLOR_BGR2RGB))  # OpenCV BGR formatında okur, bu yüzden RGB'ye dönüştürüyoruz
        plt.title(f"{i}. leveldeki görüntü ")
        plt.show()
"""