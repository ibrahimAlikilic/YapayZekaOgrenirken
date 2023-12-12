from imutils import paths
import numpy as np
import imutils
import cv2
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)
'''
    Görüntüyü gri tonlamaya çevirme (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).
    Görüntüyü yumuşatma (blurring) işlemi uygulama (cv2.GaussianBlur(gray, (5, 5), 0)).
    Kenar tespiti yapma (cv2.Canny(gray, 35, 125)).
    Kenarlar arasındaki konturları bulma (cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)).
    Konturları alıp en büyük olanı seçme (c = max(cnts, key=cv2.contourArea)).
    Seçilen konturun minimum sınırlayıcı dikdörtgenini bulma (cv2.minAreaRect(c)).
'''

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth # bu formul anlattığı link : https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24.0
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 11.0
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread("/imgRoad")
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
'''
KNOWN_DISTANCE değişkeni, kameranın nesneye olan bilinen mesafesini temsil eder. Bu örnekte, bu mesafe 24 inç olarak belirlenmiştir.

KNOWN_WIDTH değişkeni, nesnenin bilinen genişliğini temsil eder. Bu örnekte, kağıdın genişliği 11 inç olarak belirlenmiştir.

Bir önceki kod bloğunda tanımlanan find_marker fonksiyonu kullanılarak, belirli bir görüntüdeki kağıt nesnesinin minimum sınırlayıcı dikdörtgeni bulunur (marker = find_marker(image)).

KNOWN_WIDTH ve KNOWN_DISTANCE değerleri kullanılarak kamera odak uzunluğu (focalLength) hesaplanır. 
Bu hesaplama, kameranın nesnenin genişliğini algılamadaki perspektif etkilerini dikkate alır ve şu formülle yapılır:

focalLength=marker_width×known_distance/known_width

Burada, marker[1][0] ifadesi, find_marker fonksiyonu tarafından döndürülen minimum sınırlayıcı dikdörtgenin genişliğini temsil eder.

Bu kod bloğu, nesnenin gerçek dünyadaki uzaklığını daha sonra ölçmek için kullanılacak olan kamera odak uzunluğunu hesaplar. Bu tür bilgiler, özellikle kamera tabanlı uzaklık ölçümü uygulamalarında önemlidir.
'''

# loop over the images
for imagePath in sorted(paths.list_images("images")):
	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	image = cv2.imread(imagePath)
	marker = find_marker(image)
	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
	# draw a bounding box around the image and display it
	box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
	box = np.int0(box)
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fft" % (inches / 12),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	cv2.imshow("image", image)
	cv2.waitKey(0)
 
'''
paths.list_images("images"): "images" klasöründeki görüntü dosyalarını alır ve bu dosyaları sıralı bir şekilde işlemek için bir döngü başlatır.

Her bir görüntü için şu işlemleri gerçekleştirir:

Görüntüyü okur (cv2.imread(imagePath)).
find_marker fonksiyonunu kullanarak görüntüdeki kağıdın konumunu ve boyutunu belirler.
distance_to_camera fonksiyonunu kullanarak kameradan kağıda olan uzaklığı ölçer.
cv2.cv.BoxPoints ve cv2.boxPoints işlevlerini kullanarak, minimum sınırlayıcı dikdörtgenin köşe noktalarını alır ve bu noktaları bir çokgen olarak çizer 
(box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)).

Çizilen çokgeni görüntü üzerine çizer (cv2.drawContours(image, [box], -1, (0, 255, 0), 2)).
Görüntü üzerine kağıda olan uzaklığı yazan bir metin ekler 
(cv2.putText(image, "%.2fft" % (inches / 12), (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)).

İşlenen görüntüyü ekranda gösterir (cv2.imshow("image", image)).
Kullanıcıdan herhangi bir tuşa basılmasını bekler (cv2.waitKey(0)).
Bu işlemler, her bir görüntüde kağıdın konumunu ve uzaklığını belirleyerek, bu bilgileri görsel olarak vurgular ve ekranda gösterir. 
Bu tip uygulamalar, nesnelerin gerçek dünya uzaklıklarını ölçmek ve görüntü işleme uygulamalarında kullanılmak üzere kullanışlıdır.
'''