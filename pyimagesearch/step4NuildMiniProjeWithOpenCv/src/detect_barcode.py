import numpy as np
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "path to the image file")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction using OpenCV 2.4
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
#Bu satır, Sobel operatörü uygulandıktan sonraki çıkış görüntüsü için kullanılacak veri tipini belirler. 
# imutils.is_cv2() fonksiyonu ile OpenCV sürümü kontrol edilir ve veri tipi buna göre belirlenir.
gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
'''
Bu satırlar, Sobel operatörünü kullanarak giriş gri tonlamalı görüntünün (gray) gradyanını hesaplar. 
Sobel operatörü, görüntünün x ve y yönlerindeki yoğunluk değişimlerini belirleyen bir konvolüsyon işlemidir.

dx=1, dy=0 ifadesi gradX için, Sobel operatörünün x-yönünde uygulandığını belirtir.
dx=0, dy=1 ifadesi gradY için, Sobel operatörünün y-yönünde uygulandığını belirtir.
'''
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
# Bu satır, y-gradient (gradY)’i x-gradient (gradX)’den çıkarır. Sonuç, hem yatay hem de dikey yönlere göre kenarları veya yoğunluk değişimlerini temsil eder.
gradient = cv2.convertScaleAbs(gradient)
#Bu satır, gradyan görüntüsünü mutlak değerlere dönüştürür ve sonucu 8-bitlik bir görüntüye ölçekler. Bu, gradyan değerlerinin 8-bit bir görüntü için geçerli aralıkta olmasını sağlar.

# blur and threshold the image
blurred = cv2.blur(gradient, (9, 9))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
'''
Bu iki satır morfolojik kapanma işlemi uygular. Morfolojik kapanma, beyaz nesnelerin üzerindeki küçük siyah delikleri kapatmaya veya beyaz nesneleri birleştirmeye yarar.

cv2.getStructuringElement fonksiyonu, belirli bir yapısal elementi oluşturur. cv2.MORPH_RECT parametresi, dikdörtgen şeklinde bir yapısal element oluşturur. 
(21, 7) ise bu dikdörtgenin boyutunu belirtir.

cv2.morphologyEx fonksiyonu ile kapanma işlemi uygulanır. 
Bu işlem, önce beyaz bölgeleri genişletir (erozyon) ve ardından tekrar daraltır (dilation). Bu sayede beyaz nesnelerin kenarları birleştirilir ve siyah bölgeler kapatılır.

Sonuç olarak, closed görüntüsü, morfolojik kapanma işlemi uygulanmış ikili bir görüntüdür. Bu işlem genellikle nesnelerin konturlarını iyileştirmek ve küçük boşlukları kapatmak için kullanılır.
'''

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)
'''
Bu iki satır, morfolojik erozyon ve genişleme (dilation) işlemlerini serisi olarak uygular. Bu işlemler, genellikle nesne tespiti veya kontur iyileştirmesi gibi uygulamalarda kullanılır.

cv2.erode fonksiyonu, beyaz bölgeyi daraltır (erozyon). iterations=4 parametresi, erozyon işleminin kaç kez tekrarlanacağını belirtir. 
Bu, beyaz nesnelerin kenarlarını ve küçük detayları küçültmeye yardımcı olur.

cv2.dilate fonksiyonu, beyaz bölgeyi genişletir (genişleme). Yine iterations=4 parametresi, genişleme işleminin kaç kez tekrarlanacağını belirtir. 
Bu, beyaz nesnelerin kenarlarını ve küçük boşlukları doldurmaya yardımcı olur.

Bu iki işlem kombinasyonu, önce beyaz bölgeleri daraltıp ardından tekrar genişleterek, beyaz nesnelerin konturlarını ve birbirine yakın nesneler arasındaki boşlukları düzelten bir etki yaratır.
Bu genellikle nesne tespiti veya kontur tabanlı işlemlerde kullanılır.
'''
'''
All we are doing here is performing 4 iterations of erosions, followed by 4 iterations of dilations. An erosion will “erode” the white pixels in the image, thus removing the small blobs, 
whereas a dilation will “dilate” the remaining white pixels and grow the white regions back out.

Provided that the small blobs were removed during the erosion, they will not reappear during the dilation.
'''

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)
'''
Bu kod bloğu, konturlar arasında en büyük olanı bulduktan sonra, bu konturun etrafında bir döndürülmüş sınırlayıcı kutuyu (rotated bounding box) hesaplar ve bu kutunun dört köşesini elde eder. 
İşte adım adım açıklama:

cv2.minAreaRect(c): Bu satır, contours (c) içindeki en küçük alanı saran döndürülmüş bir dikdörtgenin bilgilerini içeren bir rect nesnesini hesaplar. 
Bu dikdörtgen, döndürülmüş olduğu için genellikle bir nesnenin gerçek sınırlarını daha iyi temsil eder.

cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect): Bu satır, rect nesnesinden bir dikdörtgenin köşe noktalarını alır. 
Eğer OpenCV sürümü 2.x (cv2) ise, cv2.cv.BoxPoints kullanılır; aksi takdirde, cv2.boxPoints kullanılır. Bu fonksiyon, döndürülmüş dikdörtgenin köşe noktalarını verir.

box = np.int0(box): Bu satır, köşe noktalarını integer (tam sayı) tipine dönüştürür. box değişkeni, artık bir döndürülmüş dikdörtgenin köşe noktalarını içerir.

Bu üç satır sonunda, box değişkeni, döndürülmüş bir dikdörtgenin köşe noktalarını içerir ve bu köşeler, çizilecek bir dikdörtgenin kenarlarını belirlemek için kullanılır.
'''
# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
'''
Bu dikdörtgen, genellikle nesnenin oryantasyonunu ve konumunu daha iyi temsil eden bir sınırlayıcı kutudur. 
Sonrasında, bu dikdörtgen konturu, cv2.drawContours fonksiyonu kullanılarak orijinal görüntü üzerine çizilir ve görüntü ekranda gösterilir.
'''
cv2.imshow("Image", image)
cv2.waitKey(0)