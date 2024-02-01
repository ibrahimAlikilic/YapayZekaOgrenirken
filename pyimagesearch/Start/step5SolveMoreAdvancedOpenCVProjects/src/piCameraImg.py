# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
rawCapture = PiRGBArray(camera)
'''
PiCamera sınıfından bir örnek oluşturulur ve PiRGBArray sınıfından bir raw capture (ham yakalama) nesnesi oluşturulur. 
Raw capture, kameradan alınan görüntü verisini tutar.
'''
# allow the camera to warmup
time.sleep(0.1)
# Kameranın başlatılmasından sonra, kameranın tam anlamıyla hazır olması için kısa bir bekleme süresi eklenir.

# grab an image from the camera
camera.capture(rawCapture, format="bgr")
image = rawCapture.array
'''
capture metodunu kullanarak kameradan bir görüntü yakalanır ve bu görüntü rawCapture nesnesine kaydedilir. 
Ardından, bu raw verisi array özelliği aracılığıyla bir NumPy dizisine dönüştürülerek image değişkenine atanır.
'''
# display the image on screen and wait for a keypress
cv2.imshow("Image", image)
cv2.waitKey(0)