# import the necessary packages
import numpy as np
import argparse
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "\input")
args = vars(ap.parse_args())
# load the image
image = cv2.imread(args["image2.png"])
# define the list of boundaries
boundaries = [
	([17, 15, 100], [50, 56, 200]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]
'''
Bu kod bir renk aralığını belirleyip bu aralıktaki renkleri bulan ve görselleştiren bir örnek içerir. boundaries adlı liste, renk aralıklarını belirten dört farklı renk aralığını içerir.

Her bir renk aralığı iki adet NumPy dizisi içerir: alt sınır ve üst sınır. Her bir renk sınırı, BGR (Mavi, Yeşil, Kırmızı) renk uzayında belirtilmiştir. Yani, her renk aralığı, 
bu renklerin belirli bir aralığını tanımlar.

Örneğin, [17, 15, 100] ve [50, 56, 200] renk aralığı, mavi renk tonları için bir aralığı temsil eder. Bu renk aralığı, alt sınırdaki mavi rengin (17, 15, 100) ve 
üst sınırdaki mavi rengin (50, 56, 200) olması anlamına gelir.

Her bir renk aralığı, cv2.inRange fonksiyonunda kullanılmak üzere uygun NumPy dizilere dönüştürülür. cv2.inRange fonksiyonu, 
belirtilen renk aralığına giren pikselleri beyaz, diğer pikselleri siyah yapacak bir maske oluşturur. Bu maske, ardından cv2.bitwise_and fonksiyonu ile 
orijinal görüntü üzerine uygulanır ve yalnızca belirtilen renk aralığındaki pikseller gösterilir.
'''
# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)