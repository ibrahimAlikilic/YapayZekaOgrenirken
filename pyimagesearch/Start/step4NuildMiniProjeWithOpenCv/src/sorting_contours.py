import numpy as np
import argparse
import imutils
import cv2
def sort_contours(cnts,method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse=False
    i=0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
    if method=="top-to-bottom" or method == "bottom-to-top":
        i=1
    # construct the list of bounding boxes and sort them from top to (sınırlayıcı kutuların listesini oluşturun ve bunları yukarıdan aşağıya doğru sıralayın)
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)
'''
Fonksiyon Parametreleri:

cnts: Giriş olarak alınan kontur listesi.
method: Konturların sıralanma yöntemini belirten bir parametre. Varsayılan değeri "left-to-right"tir. Diğer geçerli değerler şunlardır: "right-to-left", "top-to-bottom", "bottom-to-top".
Reverse ve İndex Değişkenleri:

reverse: Konturları ters sıralamak için bir bayrak. Varsayılan değeri False'dir.
i: Sıralama işlemi sırasında kullanılacak indeks. Başlangıçta 0 olarak ayarlanır.
Ters Sıralama Kontrolü:

Eğer sıralama yöntemi "right-to-left" veya "bottom-to-top" ise, reverse bayrağı True olarak ayarlanır.
Y Koordinatına Göre Sıralama İndexi:

Eğer sıralama yöntemi "top-to-bottom" veya "bottom-to-top" ise, i indeksi 1 olarak ayarlanır. Bu durumda konturların y koordinatına göre sıralanacaktır.
Bounding Boxes Oluşturma ve Sıralama:

boundingBoxes: Konturların sınırlayıcı kutularını temsil eden bir liste oluşturulur.
Konturlar ve sınırlayıcı kutular, zip fonksiyonu kullanılarak birleştirilir ve y koordinatına veya x koordinatına göre sıralamak için sorted fonksiyonu kullanılır.
Sıralama yöntemine göre sıralama işlemi gerçekleştirilir ve reverse bayrağına göre ters sıralama yapılır.
Sonuçları Döndürme:

Sıralanmış konturlar ve sınırlayıcı kutular, ayrı ayrı döndürülür.
'''
def draw_contour(image, c, i):
	# compute the center of the contour area and draw a circle
	# representing the center
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	# draw the countour number on the image
	cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)
	# return the image with the contour number drawn on it
	return image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-m", "--method", required=True, help="Sorting method")
args = vars(ap.parse_args())
# load the image and initialize the accumulated edge image
image = cv2.imread(args["image"])
accumEdged = np.zeros(image.shape[:2], dtype="uint8")
# loop over the blue, green, and red channels, respectively
for chan in cv2.split(image):
	# blur the channel, extract edges from it, and accumulate the set
	# of edges for the image
	chan = cv2.medianBlur(chan, 11)
	edged = cv2.Canny(chan, 50, 200)
	accumEdged = cv2.bitwise_or(accumEdged, edged)
# show the accumulated edge map
cv2.imshow("Edge Map", accumEdged)
'''
Bu Python betiği, argparse kütüphanesini kullanarak komut satırı argümanlarını işleyen ve bir görüntüde kenar tespiti uygulayan basit bir örnek içerir. Ayrıntılı bir açıklama şu şekildedir:

Argparse Tanımlama:

argparse.ArgumentParser() ile bir argüman analizleyici (parser) oluşturulur.
add_argument fonksiyonu ile komut satırından alınacak argümanlar tanımlanır.
-i veya --image: Giriş görüntüsünün dosya yolu. Zorunlu bir argümandır.
-m veya --method: Konturları sıralama yöntemi. Zorunlu bir argümandır.
Argümanları Ayrıştırma:

ap.parse_args() ile komut satırından gelen argümanlar ayrıştırılır.
args adlı bir sözlük, ayrıştırılmış argüman değerlerini içerir.
Görüntüyü Yükleme ve Kenar Haritası İçin Hazırlık:

cv2.imread(args["image"]) ile komut satırından alınan görüntü yüklenir.
np.zeros(image.shape[:2], dtype="uint8") ile aynı boyutta bir sıfırlardan oluşan bir matris (accumEdged) oluşturulur.
Kanal Üzerinde Döngü:

cv2.split(image) ile görüntü kanallarına ayrılır (mavi, yeşil, kırmızı).
Her bir kanal üzerinde aşağıdaki işlemler gerçekleştirilir:
cv2.medianBlur(chan, 11): Medyan bulanıklaştırma uygulanır.
cv2.Canny(chan, 50, 200): Canny kenar tespiti uygulanır.
Elde edilen kenar haritası, accumEdged üzerinde birleştirilir (cv2.bitwise_or).
Toplanmış Kenar Haritasını Gösterme:

cv2.imshow("Edge Map", accumEdged) ile elde edilen toplu kenar haritası görselleştirilir.
cv2.waitKey(0) ve cv2.destroyAllWindows() ile beklenir ve pencere kapatılır.
Bu betik, giriş görüntüsünü kanallara ayırarak her bir kanal üzerinde medyan bulanıklaştırma ve Canny kenar tespiti uygular. 
Sonuç olarak, her kanalın kenar haritaları birleştirilir ve toplu bir kenar haritası elde edilir. Bu süreç, görüntüdeki önemli kenarları vurgulamak için kullanılabilir.
'''

# find contours in the accumulated image, keeping only the largest
# ones
cnts = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
'''
cv2.findContours: Kenar haritasındaki konturları bulur.
imutils.grab_contours: OpenCV sürümleri arasındaki uyumluluk sorunlarını ele alarak konturları çeker.
sorted: Konturları alanlarına göre sıralar, büyükten küçüğe doğru sıralama yapar.
[:5]: Yalnızca en büyük 5 konturu alır.
'''
orig = image.copy()
# loop over the (unsorted) contours and draw them
for (i, c) in enumerate(cnts):
	orig = draw_contour(orig, c, i)
# show the original, unsorted contour image
cv2.imshow("Unsorted", orig)
# sort the contours according to the provided method
(cnts, boundingBoxes) = sort_contours(cnts, method=args["method"]) # Önceki kısımda tanımlanan sort_contours fonksiyonunu kullanarak konturları belirli bir yönteme göre sıralar.
# loop over the (now sorted) contours and draw them
for (i, c) in enumerate(cnts):
	draw_contour(image, c, i)
# show the output image
cv2.imshow("Sorted", image)
cv2.waitKey(0)