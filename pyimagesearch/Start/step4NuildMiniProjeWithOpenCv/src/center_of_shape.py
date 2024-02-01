import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())
# load the image, convert it to grayscale, blur it slightly,
# and threshold it
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
'''[1] in manasını hatırlamadım ve chatgit ye sordum , cevabı :
Gri tonlamalı görüntü üzerinde eşikleme işlemi yapmaktadır. Fonksiyonu ve [1] ifadesinin anlamını şu şekilde açıklayabiliriz:

cv2.threshold: Bu fonksiyon, giriş görüntüsündeki her piksel üzerinde sabit bir seviyedeki eşikleme işlemi uygulamak için kullanılır. Belirli bir eşik değerinden büyük olan pikseller, belirli bir maksimum değere (bu durumda 255) ayarlanır ve eşik değerinden küçük veya eşit olan pikseller sıfıra ayarlanır.

blurred: Bulanık gri tonlamalı görüntü, eşikleme işleminin uygulandığı giriş.

60: Bu, eşik değeridir. 60'tan büyük olan pikseller 255'e ayarlanacak ve 60'a eşit veya küçük olan pikseller sıfıra ayarlanacaktır.

255: Bu, eşik değerinden yüksek olan piksellere atanacak maksimum değerdir.

cv2.THRESH_BINARY: Bu, uygulanan eşikleme türünü belirtir. Bu durumda, bir pikselin yoğunluğu eşik değerinden büyükse, maksimum değere (255) ayarlanır, aksi takdirde sıfıra ayarlanır.

Satırdaki [1], cv2.threshold fonksiyonunun döndürdüğü demetin ikinci elemanına erişmek için kullanılır. Fonksiyon, iki elemanlı bir demet döndürür: kullanılan eşik değeri (bu durumda kullanılmaz ve genellikle _ ile adlandırılır) ve eşiklenmiş görüntü. [1] kullanarak kod, demetten eşiklenmiş görüntüyü çıkartır ve thresh değişkenine atar.

Özetle, bu satır, bulanık görüntüye ikili eşikleme uygulamakta ve [1] kullanılarak eşiklenmiş görüntü demetten çıkartılarak thresh değişkenine atanmaktadır.
'''
# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
''' ezberimdeydi ama anlamlarını tam hatırlayamadım ve bu sebepten chatgpt ye sordum , cevabı :
Bu kod, OpenCV ve imutils kütüphanelerini kullanarak kontur (sınırlayıcı çizgi) tespiti işlemlerini gerçekleştiriyor. İşte kodun bu bölümünün ayrıntıları:

cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE):

thresh.copy(): Bu, eşiklenmiş görüntünün bir kopyasını oluşturur. Kontur tespiti işlemi sırasında orijinal görüntünün değişmemesi için bu kopya kullanılır.
cv2.RETR_EXTERNAL: Bu, konturların sadece dış konturlarını almayı belirtir. Yani, iç içe geçmiş konturların içindeki konturları almaz.
cv2.CHAIN_APPROX_SIMPLE: Bu, konturun temsilini daha basitleştirmek için kullanılır. Örneğin, bir çizginin ardışık pikselleri arasındaki bağlantıyı temsil etmek için sadece uç noktaları saklar.
cnts = imutils.grab_contours(cnts):

imutils.grab_contours fonksiyonu, OpenCV'nin farklı sürümleri arasındaki uyumsuzlukları ele almak için tasarlanmış bir yardımcı işlemdir. Bu durumda, cv2.findContours işlevinin döndürdüğü değerlerin uygun bir şekilde elde edilmesini sağlar.
cnts değişkeni, kontur tespiti işleminden elde edilen konturları içerir. cnts değişkeni, farklı OpenCV sürümlerinde uyumluluk sağlamak için kullanılan imutils fonksiyonu tarafından düzenlenmiştir.
Bu kod parçasının genel amacı, belirli bir eşik değeri uygulanmış görüntü üzerinde kontur tespiti yapmaktır. Elde edilen konturlar daha sonra çeşitli amaçlar için kullanılabilir, örneğin nesne tanıma, nesne sınırlarını çizme veya nesne alanını hesaplama gibi görsel işleme görevlerinde.
'''
# loop over the contours
for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
    # cv2.moments fonksiyonu, bir konturun momentini hesaplar. Bu momentler, konturun ağırlık merkezini hesaplamak için kullanılır. 
    # cX ve cY, konturun x ve y koordinatlarındaki ağırlık merkezini temsil eder.
	# draw the contour and center of the shape on the image
 
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
	cv2.putText(image, "center", (cX - 20, cY - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    #cv2.drawContours: Bu fonksiyon, konturu orijinal görüntü üzerine çizer. Renk olarak (0, 255, 0) yani yeşil kullanılmış ve çizgi kalınlığı 2 olarak belirlenmiştir.
    #cv2.circle: Bu fonksiyon, konturun ağırlık merkezini içeren bir daire çizer. Bu dairenin rengi (255, 255, 255) yani beyaz, ve çizgi kalınlığı -1 olduğu için içi dolu bir daire olacaktır.
    #cv2.putText: Bu fonksiyon, belirli bir metni görüntüye ekler. Burada "center" metni, ağırlık merkezi etrafında görüntüye eklenir.
	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)