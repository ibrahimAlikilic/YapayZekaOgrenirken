# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
'''
Bu bölümde, PiCamera sınıfından bir kamera nesnesi oluşturulur. 
Ardından, bu kamera nesnesinin çözünürlüğü (resolution) 640x480 piksel olarak ayarlanır ve kamera çerçeve hızı (framerate) 32 kare/saniye olarak ayarlanır.
'''
rawCapture = PiRGBArray(camera, size=(640, 480))
# PiRGBArray sınıfından bir raw capture nesnesi oluşturulur ve bu nesne, kameradan alınacak olan raw veriyi (ham görüntü verisi) tutar. Ayrıca, çözünürlük bilgisi burada da belirtilir.
# allow the camera to warmup
time.sleep(0.1)
# Kameranın başlatılmasından sonra, kameranın tam anlamıyla hazır olması için kısa bir bekleme süresi eklenir.
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	'''
	Bu döngü, kameradan sürekli olarak görüntü yakalamak için kullanılır. capture_continuous metodu, sürekli olarak kareler alır ve rawCapture nesnesine kaydeder. 
	use_video_port=True parametresi, daha hızlı video modunu kullanmaya olanak tanır.
	'''
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	image = frame.array
	# show the frame
	cv2.imshow("Frame", image)
	'''
	Her karede, frame değişkeninden NumPy dizisi olan array özelliği aracılığıyla bir görüntü alınır ve bu görüntü OpenCV kullanılarak ekranda gösterilir.
	'''
	key = cv2.waitKey(1) & 0xFF
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
	# Her kare alındıktan sonra, rawCapture nesnesi temizlenir ve bir sonraki kare için hazırlanır. Bu, önceki karelerin bellekte birikmesini engeller.
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break