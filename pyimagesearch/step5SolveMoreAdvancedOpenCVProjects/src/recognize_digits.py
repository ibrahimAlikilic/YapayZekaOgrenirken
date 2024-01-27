# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
	(1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}
'''
Bu kısımda, DIGITS_LOOKUP adında bir sözlük tanımlanıyor. 
Bu sözlüğün anahtarları, 7-segment ekranındaki segmentlerin konfigürasyonunu temsil eden demetlerdir ve değerleri ilgili rakamlardır. 
Bu sözlük, bir görüntüde tespit edilen segmentleri ilgili rakamlara eşleştirmek için kullanılır.
'''
# load the example image
image = cv2.imread("input/example.png")
# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(blurred, 50, 200, 255)
'''
Bu satır, kenar belirleme işlemi uygular. cv2.Canny fonksiyonu, kenarları tespit etmek için Canny kenar tespiti algoritmasını kullanır. Parametreler şu şekildedir:

    blurred: Bulanıklaştırılmış gri tonlamalı görüntü.
    50: Kenar tespiti için düşük eşik değeri.
    200: Kenar tespiti için yüksek eşik değeri.
    255: Kenar piksellerinin beyaz olarak işaretlenmesi için kullanılan değer.

Sonuç olarak, edged adlı değişken, orijinal görüntünün kenarlarını temsil eden bir ikili (siyah-beyaz) görüntüdür. 
Bu kenarlar daha sonra kontur tespiti için kullanılabilir.
'''
# find contours in the edge map, then sort them by their
# size in descending order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts) # Bu satır, imutils kütüphanesindeki grab_contours fonksiyonunu kullanarak OpenCV 2 ve 3 arasındaki uyumsuzlukları ele alır ve doğru kontur verilerini elde eder.
cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # Bu satır, konturları, kontur alanlarına göre sıralar. Büyükten küçüğe doğru sıralama yapılır, bu sayede en büyük kontur, muhtemelen termostat ekranını temsil eden kontur olacaktır.
displayCnt = None # Bu satırda, termostat ekranının konturunu temsil eden değişken displayCnt başlangıçta None (hiçbir şey) olarak atanır.
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if the contour has four vertices, then we have found
	# the thermostat display
	if len(approx) == 4:
		displayCnt = approx
		break
	'''
	Bu bölüm, konturlar üzerinde bir döngü oluşturur. Her kontur için bir yaklaşım (approximation) yapılır. cv2.arcLength fonksiyonu, konturun uzunluğunu hesaplar. 
	cv2.approxPolyDP fonksiyonu ise belirtilen yüzde oranında yaklaşım yaparak konturu basitleştirir.

    Eğer yaklaşık dört kenarı olan bir kontur bulunursa (len(approx) == 4), bu, muhtemelen termostat ekranının konturunu temsil eder. 
	Bu durumda, displayCnt değişkenine bu kontur atanır ve döngü sona erer.
    '''
# extract the thermostat display, apply a perspective transform
# to it
warped = four_point_transform(gray, displayCnt.reshape(4, 2))
'''
Bu satır, imutils.perspective.four_point_transform fonksiyonunu kullanarak perspektif dönüşümü uygular. 
gray adlı gri tonlamalı görüntü üzerinde bu dönüşümü uygulamak, perspektif bozulmalarını düzeltmeye ve termostat ekranını düzleştirmeye yöneliktir. 
displayCnt.reshape(4, 2) ifadesi, termostat ekranını temsil eden dörtgenin köşe noktalarını içerir.
'''
output = four_point_transform(image, displayCnt.reshape(4, 2))
'''
Bu satır, orijinal renkli görüntü üzerinde de aynı perspektif dönüşümünü uygular. 
Bu, orijinal görüntünün perspektif bozulmalarını düzelten ve termostat ekranını düzleştiren bir işlemdir. Elde edilen düzeltilmiş görüntü output değişkenine atanır.

Bu adımların sonucunda, warped ve output değişkenleri, termostat ekranını temsil eden dörtgenin perspektif dönüşümü sonucu elde edilen gri tonlamalı ve 
renkli görüntülerdir. Bu işlemler, daha sonra rakamları tanımak için yapılacak adımlar için görüntünün hazırlanmasına yöneliktir.
'''
# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
# Bu satır, morfolojik işlemler için kullanılacak çekirdek (kernel) elemanını oluşturur. 
# cv2.MORPH_ELLIPSE ile elips şeklinde bir çekirdek seçilir ve (1, 5) boyutları belirlenir.
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
'''
Bu satır, açma (opening) morfolojik işlemi uygular. Açma işlemi, nesneler arasındaki boşlukları genişletir ve küçük nesneleri kaldırarak görüntüyü temizler. 
cv2.MORPH_OPEN parametresi, açma işlemi için kullanılır, ve kernel çekirdeği bu işleme uygulanır.
'''
# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []
# loop over the digit area candidates
for c in cnts:
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)
	# if the contour is sufficiently large, it must be a digit
	if w >= 15 and (h >= 30 and h <= 40):
		digitCnts.append(c)
		'''
		Bu bölüm, konturlar üzerinde bir döngü oluşturur. Her konturun sınırlayıcı kutusunu hesaplar (cv2.boundingRect). 
		Eğer bu sınırlayıcı kutu, belirli bir genişlik (15 piksel) ve yükseklik (30 ila 40 piksel aralığında) koşullarını sağlıyorsa, 
		bu konturun muhtemelen bir rakamı temsil ettiği düşünülerek digitCnts listesine eklenir.

        Sonuç olarak, digitCnts listesi, görüntüdeki muhtemel rakamları temsil eden konturları içerir. Bu konturlar daha sonra rakamları tanımak için kullanılabilir.
        '''
# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts,
	method="left-to-right")[0]
'''
    contours.sort_contours fonksiyonu, konturları sıralamak için kullanılır.
    digitCnts listesi, sıralı konturları içeren bir demetle güncellenir.
    method="left-to-right" parametresi, konturların sola doğru sıralanmasını sağlar.
'''
digits = []
# loop over each of the digits
for c in digitCnts:
	# extract the digit ROI
	(x, y, w, h) = cv2.boundingRect(c)
	roi = thresh[y:y + h, x:x + w]
	
	# compute the width and height of each of the 7 segments
	# we are going to examine
	(roiH, roiW) = roi.shape
	(dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
	dHC = int(roiH * 0.05)
	# Bu bölümde, her bir 7 segmentin genişliği ve yüksekliği hesaplanır. Bu değerler, ilerleyen adımlarda her bir segmentin bölgesini belirlemede kullanılır.
	
	# define the set of 7 segments
	segments = [
		((0, 0), (w, dH)),	# top
		((0, 0), (dW, h // 2)),	# top-left
		((w - dW, 0), (w, h // 2)),	# top-right
		((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
		((0, h // 2), (dW, h)),	# bottom-left
		((w - dW, h // 2), (w, h)),	# bottom-right
		((0, h - dH), (w, h))	# bottom
	]
	# Bu bölümde, her bir rakamın üzerindeki 7 segmentin koordinatları belirlenir. Her bir segment, iki nokta arasındaki dikdörtgen bölgeyi temsil eder.
	on = [0] * len(segments)
	'''
	Bu satır, her bir segmentin yanıp sönme durumunu takip eden bir liste oluşturur. 
	Başlangıçta, hiçbir segment yanmıyor (on listesi sıfırlarla dolu). 
	Yanıp sönme durumları, her bir segmentin içindeki rakamın hangi segmentleri içerdiğini belirlemek için kullanılacak.
	'''
		# loop over the segments
	for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
		# extract the segment ROI, count the total number of
		# thresholded pixels in the segment, and then compute
		# the area of the segment
		segROI = roi[yA:yB, xA:xB]
		total = cv2.countNonZero(segROI)
		area = (xB - xA) * (yB - yA)
		# if the total number of non-zero pixels is greater than
		# 50% of the area, mark the segment as "on"
		if total / float(area) > 0.5:
			on[i]= 1
			'''
			Bu bölümde, her bir segmentin ROI'sini (Region of Interest) çıkarır, bu segmentteki eşiklenmiş piksellerin toplam sayısını ve segmentin alanını hesaplar. 
			Eğer segmentteki toplam piksel sayısı, segment alanının %50'sinden fazlaysa, bu segmenti "açık" olarak işaretler (on[i]= 1).
			'''
	# lookup the digit and draw it on the image
	digit = DIGITS_LOOKUP[tuple(on)]
	digits.append(digit)
	cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
	cv2.putText(output, str(digit), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
	'''
	Bu bölümde, segmentlerin durumlarına göre, yani hangi segmentlerin açık veya kapalı olduğuna göre bir rakam belirlenir. 
	Belirlenen rakam, DIGITS_LOOKUP sözlüğünden alınır ve digits listesine eklenir. 
	Ayrıca, orijinal görüntü üzerine, belirlenen rakamı içeren bir dikdörtgen çizilir ve rakam metni eklenir.
	'''
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)