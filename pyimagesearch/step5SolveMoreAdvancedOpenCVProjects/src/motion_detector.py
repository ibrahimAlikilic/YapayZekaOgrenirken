# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
'''
Bu bölüm, komut satırı argümanlarını işlemek için bir argüman ayrıştırıcı oluşturuyor. 
Kodun çalıştırılabilir hale getirilmesi için bu argümanlardan biri -v veya --video ile belirtilen video dosyasının yolu olabilir. 
Diğer bir argüman -a veya --min-area, algılanabilir hareketlerin minimum alanını belirtir.
'''
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
# Bu bölüm, eğer -v veya --video argümanı belirtilmemişse, bir webcam'den (src=0) video akışını başlatır. 
# Eğer argüman belirtilmişse, belirtilen video dosyasını açar.
# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])
# initialize the first frame in the video stream
firstFrame = None
'''
Bu satır, hareket algılama algoritmasının çalışması için bir temel kare (frame) başlatır. 
İlk kare, ilerleyen adımlarda mevcut kare ile karşılaştırılacak ve değişiklik tespit edilecek olan referans bir kare olarak kullanılacaktır. 
İlk başta bu kare None olarak atanır.

Bu kod örneği bir hareket algılama uygulamasıdır ve birçok bilgisayar görüşü uygulamasında kullanılan temel kavramları içerir. H
areket algılama, bir kameradan alınan görüntülerdeki değişiklikleri tespit ederek belirli bir olayın gerçekleşip gerçekleşmediğini kontrol etmeye yarar. 
Bu örnekte, bir video dosyası veya webcam kullanılarak alınan görüntüler üzerindeki hareket algılanmaya çalışılmaktadır.
'''
# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	text = "Unoccupied"
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if frame is None:
		break
	'''
	Bu bölüm, bir video akışındaki her kareyi işlemek üzere sonsuz bir döngü oluşturur. 
	Her döngüde, bir sonraki kare (frame) alınır. Eğer video dosyası kullanılıyorsa ve kare alınamıyorsa (frame değeri None ise), döngüyü sonlandırır.
	'''
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	'''
	Bu kısım, her kareyi işlemeden önce bazı ön işlemler uygular.
	İlk olarak, kare boyutunu 500 piksel genişlikte bir hale getirir (imutils.resize fonksiyonu kullanılarak). 
	Ardından, kareyi gri tona dönüştürür (cv2.cvtColor) ve gürültüyü azaltmak için bir Gauss filtresi uygular (cv2.GaussianBlur).
	'''
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue
	'''
	Bu bölümde, eğer firstFrame (ilk referans kare) henüz belirlenmemişse (yani None ise), 
	şu andaki kareyi (gray) ilk referans kare olarak kullanır ve döngüye geri döner.
	'''
	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	'''
	Bu kısım, ilk referans kare (firstFrame) ile şu anki kare (gray) arasındaki mutlak farkı hesaplar (cv2.absdiff). 
	Ardından, bu fark karesini belirli bir eşik değeri üzerinden ikili bir görüntüye dönüştürür (cv2.threshold). 
	Bu eşik değeri (25), fark görüntüsünde 25'ten büyük olan pikselleri beyaz, küçük olanları siyah yapar.
	'''
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	'''
	Bu kısım, eşiklenmiş görüntüdeki delikleri doldurmak için bir dilate işlemi uygular (cv2.dilate).
	Ardından, görüntüdeki konturları bulur (cv2.findContours). imutils.grab_contours fonksiyonu, farklı OpenCV sürümleri arasındaki uyumsuzlukları düzeltir 
	ve konturları almak için kullanılır.
	'''
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"
		'''
		Bu bölümde, konturlar üzerinde döngüye girilir. 
		Her bir konturun alanı (cv2.contourArea(c)) belirlenen minimum alandan (args["min_area"]) büyükse, bu konturun bir nesneyi temsil ettiği kabul edilir. 
		Bu durumda, nesnenin etrafına bir dikdörtgen çizilir (cv2.rectangle) ve text değişkeni "Occupied" olarak güncellenir.
		'''
	# draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
'''
Bu bölümde, video akışını veya video dosyasını serbest bırakmak ve açık olan tüm pencereleri kapatmak için gerekli temizleme işlemleri yapılır. 
vs.stop() fonksiyonu, eğer webcam kullanılıyorsa, VideoStream'ı durdurur. Aksi halde, vs.release() fonksiyonu video dosyasını serbest bırakır. cv2.destroyAllWindows() fonksiyonu ise tüm açık pencereleri kapatır.
'''