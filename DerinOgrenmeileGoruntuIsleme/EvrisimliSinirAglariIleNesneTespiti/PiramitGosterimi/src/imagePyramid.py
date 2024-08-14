import cv2
import matplotlib.pyplot as plt

def imagePyramid(image, scale=1.5, min_size=(224,224)):
    yield image
    while True:
        w=int(image.shape[1]/scale)
        image=cv2.resize(image,dsize=(w,w))
        
        if image.shape[0]<min_size[1] or image.shape[1]<min_size[0]: # minimum boyuta ulaşana kadar devam et
            break
        yield image
"""
bir görüntü piramidi oluşturmak için kullanılan bir fonksiyon tanımlıyor. 
Bu fonksiyon, başlangıç görüntüsünden başlayarak belirli bir ölçek faktörü kullanarak görüntüyü yeniden boyutlandırır ve belirli bir minimum boyuta ulaşana kadar 
bu işlemi devam ettirir. Bu tür bir görüntü piramidi, nesne tespiti gibi uygulamalarda yararlı olabilir. Fonksiyonun ayrıntılarına bakalım:
1. Parametreler

    image: İşlem yapılacak olan başlangıç görüntüsü.
    scale: Her adımda görüntü boyutunu küçültmek için kullanılan ölçek faktörü. Bu durumda, her adımda görüntünün boyutu scale değeri ile bölünecektir. 
    Varsayılan olarak 1.5.
    
    min_size: Görüntünün boyutunun küçültülmesinden sonra ulaşması gereken minimum boyut. 
    Bu durumda (224, 224) olarak ayarlanmış, yani genişlik ve yükseklik en az 224 piksel olmalıdır.

2. İşleyiş

    İlk yield image: İlk olarak, başlangıç görüntüsü yield anahtar kelimesi ile üretilir. 
    Bu, fonksiyonun bir jeneratör işlevi görmesini sağlar ve başlangıçta mevcut görüntüyü döndürür.

    w=int(image.shape[1]/scale): Görüntünün yeni genişliği scale faktörüne bölünerek hesaplanır. Burada image.shape[1] görüntünün mevcut genişliğidir.
    w değişkeni bu genişliğin scale ile bölünmesiyle elde edilir.

    image=cv2.resize(image,dsize=(w,w)): OpenCV kütüphanesi kullanılarak, görüntü yeni genişliğe (w) göre yeniden boyutlandırılır. Bu, görüntünün boyutunu küçültür.

    if image.shape[0]<min_size[1] or image.shape[1]<min_size[0]: Görüntünün boyutlarının minimum boyutlardan biriyle karşılaştırılması yapılır.
    Eğer genişlik (image.shape[1]) veya yükseklik (image.shape[0]) minimum boyutların altındaysa, döngüden çıkılır.

    break: Eğer yukarıdaki koşul sağlanıyorsa, döngü kırılır ve fonksiyon sonlandırılır.

    Son yield image: Son olarak, son boyutlandırılmış görüntü yield anahtar kelimesi ile döndürülür.

Özet

Fonksiyon, başlangıç görüntüsünden başlayarak her adımda belirli bir ölçek faktörü kullanarak görüntüyü küçültür.
Küçültme işlemi, görüntü belirli bir minimum boyutun altına düşene kadar devam eder.
Fonksiyon, her adımda görüntüyü döndürür, bu nedenle bir görüntü piramidi elde edilir.
"""
img=cv2.imread("../input/husky.jpg")
im=imagePyramid(img,1.5,(10,10))
for i,image in enumerate(im):
    print(f"{i}. resim oluşturuldu")
    # bir tanesini görelim
    if i == 7: # i yi arttırdıkça netlik azalmış halini görüyoruz
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV BGR formatında okur, bu yüzden RGB'ye dönüştürüyoruz
        plt.title(f"{i}. leveldeki görüntü ")
        plt.show()