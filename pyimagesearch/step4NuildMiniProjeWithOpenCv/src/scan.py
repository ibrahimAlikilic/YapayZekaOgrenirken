# bunları aldığım link : https://pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
# alttakileri önceden oluşturmuş olmak lazımdı ama ben orayı kaçırdım o yüzden burada yaptıklarımın çıktısını görmek için link bıraktım.
from transform import four_point_transform # bu var bunu geçen hafta yaptık deyip link koymuş oradan yaptım .
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
'''
İlk olarak imajımızı diskten 17. Satıra yüklüyoruz .

Görüntü işlemeyi hızlandırmanın yanı sıra kenar algılama adımımızı daha doğru hale getirmek için, taranan görüntümüzü 17-20. Satırlarda 500 piksel yüksekliğe sahip olacak şekilde yeniden boyutlandırıyoruz .

Ayrıca takip etmeye de özel dikkat gösteriyoruz.ratio görüntünün orijinal yüksekliğinin yeni yüksekliğe oranı ( Satır 18 ) — bu, taramayı yeniden boyutlandırılan görüntü yerine orijinal görüntü üzerinde gerçekleştirmemize olanak tanır .

Buradan, Satır 24'te görüntüyü RGB'den gri tonlamaya dönüştürürüz , yüksek frekanslı gürültüyü gidermek için Gauss bulanıklaştırma gerçekleştiririz (2. Adımda kontur tespitine yardımcı olur) ve Satır 26'da Canny kenar tespiti gerçekleştiririz .

Adım 1'in çıktısı daha sonra Satır 30 ve 31'de gösterilir .
'''

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True) #anlamak için bulduğum site : https://jn7.net/python-ve-opencv-ile-basit-sekil-algilama-uygulamasi/
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
'''
Yapmayı sevdiğim güzel bir performans tüyosu aslında konturları alana göre sıralamak ve yalnızca en büyük olanları tutmaktır ( Satır 39 ). Bu bize konturların yalnızca en büyüğünü inceleyerek geri kalanını göz ardı etmemizi sağlar.

Daha sonra 42. Çizgi üzerindeki konturlar üzerinde döngü yapmaya başlıyoruz ve 44. ve 45. Çizgi üzerindeki noktaların sayısını yaklaşık olarak hesaplıyoruz .

Yaklaşık konturun dört noktası varsa ( Çizgi 49 ), görüntüdeki belgeyi bulduğumuzu varsayarız.

Ve yine söylüyorum, bu oldukça güvenli bir varsayımdır. Tarayıcı uygulaması (1) taranacak belgenin görüntünün ana odağı olduğunu ve (2) belgenin dikdörtgen olduğunu ve dolayısıyla dört farklı kenara sahip olacağını varsayacaktır.

Oradan 55. ve 56. satırlar taramaya gittiğimiz belgenin dış hatlarını gösteriyor.
'''

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows() # kodda yoktu ama ben olması gerekiyor diye düşündüğümden ekledim
'''
İki argümanı aktaracağızfour_point_transform: ilki, diskten yüklediğimiz orijinal görüntümüzdür ( yeniden boyutlandırılan değil ) ve ikinci argüman, belgeyi temsil eden konturun yeniden boyutlandırma oranıyla çarpımıdır.

Peki neden yeniden boyutlandırma oranıyla çarptığımızı merak ediyor olabilirsiniz?

Yeniden boyutlandırma oranıyla çarpıyoruz çünkü kenar tespiti yaptık ve yeniden boyutlandırılan görüntü üzerinde yükseklik=500 piksel konturlar bulduk.

Ancak taramayı yeniden boyutlandırılan görüntü üzerinde değil orijinal görüntü üzerinde yapmak istiyoruz , bu nedenle kontur noktalarını yeniden boyutlandırma oranıyla çarpıyoruz.

Görüntüde siyah beyaz hissini elde etmek için çarpık görüntüyü alıp gri tonlamaya dönüştürüyoruz ve 66-68 numaralı satırlara uyarlanabilir eşikleme uyguluyoruz .

Son olarak çıktımızı 72-74. Satırlarda gösteriyoruz .
'''
