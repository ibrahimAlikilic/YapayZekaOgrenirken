# import the necessary packages
from pyimagesearch.tempimage import TempImage
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import dropbox
import imutils
import json
import time
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="path to the JSON configuration file")
args = vars(ap.parse_args())
'''
argparse kütüphanesi kullanılarak bir argüman analizleyici (argument parser) oluşturulur. 
-c veya --conf bayraklarıyla belirtilen yapılandırma dosyasının yolu (path) zorunlu bir argümandır (required=True). ap.parse_args() metodu, 
komut satırından girilen argümanları analiz eder ve args adlı bir sözlük içinde saklar.
'''
# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None
'''
warnings.filterwarnings("ignore") satırı, Python tarafından üretilen uyarıları görmezden gelmek için kullanılır. 
Bu, programın çalıştığı sırada ortaya çıkabilecek bazı uyarıları bastırmak için eklenmiş olabilir.

Ardından, json.load(open(args["conf"])) satırı, argüman olarak verilen JSON formatındaki yapılandırma dosyasını okur ve conf adlı bir sözlüğe yükler. 
Bu dosya, programın çalışma zamanında kullanılacak olan çeşitli ayarları içerir.

client = None satırı, Dropbox istemcisini temsil edecek bir değişkeni başlangıçta None olarak ayarlar. 
Bu değişkenin daha sonra programın geri kalan kısmında kullanılması bekleniyor.
'''
# check to see if the Dropbox should be used
if conf["use_dropbox"]:
	'''
	Bu if ifadesi, yapılandırma dosyasındaki "use_dropbox" anahtarının değerini kontrol eder.
    Eğer "use_dropbox" değeri True ise, if bloğu çalıştırılır; yani Dropbox entegrasyonu kullanılacak demektir.
	'''
	# connect to dropbox and start the session authorization process
	client = dropbox.Dropbox(conf["dropbox_access_token"])
	'''
	Dropbox entegrasyonu için bir Dropbox istemcisini başlatır.
    conf["dropbox_access_token"] ifadesi, yapılandırma dosyasındaki "dropbox_access_token" anahtarının değerini temsil eder. 
	Bu, Dropbox API erişim belirtecini içerir.
	'''
	print("[SUCCESS] dropbox account linked")
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))
# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0
'''
camera = PiCamera(): Raspberry Pi kamera modülünü başlatır.

camera.resolution = tuple(conf["resolution"]) ve camera.framerate = conf["fps"]: Kamera çözünürlüğü ve çerçeve hızını, yapılandırma dosyasındaki "resolution" ve "fps" değerlerine göre ayarlar.

rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"])): Kameradan alınan ham görüntüyü işlemek için bir PiRGBArray nesnesi oluşturur. Bu nesne, belirtilen çözünürlükte bir görüntüyü saklar.

time.sleep(conf["camera_warmup_time"]): Kameranın hazırlanması için belirtilen süre kadar bekler. Bu süre, kameranın başlamasından önce geçen süredir ve genellikle kameranın istikrarlı bir şekilde çalışması için bir bekleme süresidir.

avg = None, lastUploaded = datetime.datetime.now(), motionCounter = 0: Ortalama kare, son yüklenen zaman damgası ve hareket sayacını başlatır. Bu değişkenler, hareket algılama ve görüntü işleme işlemleri sırasında kullanılır. avg özellikle, kameradan gelen görüntülerin ortalamasını tutarak hareket algılamasını gerçekleştirmek için kullanılır.
'''
# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image and initialize
	# the timestamp and occupied/unoccupied text
	frame = f.array
	timestamp = datetime.datetime.now()
	text = "Unoccupied"
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	# if the average frame is None, initialize it
	if avg is None:
		print("[INFO] starting background model...")
		avg = gray.copy().astype("float")
		rawCapture.truncate(0)
		continue
	# accumulate the weighted average between the current frame and
	# previous frames, then compute the difference between the current
	# frame and running average
	cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
	'''
	    for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        Bu for döngüsü, kameradan sürekli olarak kareleri yakalamak için kullanılır. capture_continuous metodu, kameradan ardışık kareler alır ve bu kareleri rawCapture nesnesine kaydeder.

    frame = f.array, timestamp = datetime.datetime.now(), text = "Unoccupied":
        Güncel kare, zaman damgası ve başlangıçta "Unoccupied" olarak ayarlanmış bir metin oluşturulur.

    frame = imutils.resize(frame, width=500):
        Görüntüyü yeniden boyutlandırır. Bu, daha küçük bir görüntü üzerinde çalışmayı kolaylaştırabilir.

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ve gray = cv2.GaussianBlur(gray, (21, 21), 0):
        Görüntüyü gri tonlamaya çevirir ve ardından bir Gauss (blurlama) filtresi uygular. Bu, daha sonra kullanılacak olan hareket algılama işlemleri için görüntüyü hazırlar.

    if avg is None::
        Eğer ortalama kare (avg) henüz başlatılmamışsa, bu bölümde başlatılır ve bir sonraki kare için döngü devam ettirilir.

    cv2.accumulateWeighted(gray, avg, 0.5) ve frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg)):
        Eğer ortalama kare varsa, cv2.accumulateWeighted metodu ile ağırlıklı ortalamaya güncel kare eklenir ve ardından cv2.absdiff metodu ile güncel kare ile ağırlıklı ortalama arasındaki fark (frameDelta) hesaplanır. Bu fark, hareket algılama işlemi için kullanılacaktır.
		'''
		# threshold the delta image, dilate the thresholded image to fill
	# in holes, then find contours on thresholded image
	thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
		cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < conf["min_area"]:
			continue
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"
	# draw the text and timestamp on the frame
	ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)
	'''
	thresh = cv2.dilate(thresh, None, iterations=2):

    Eşiklenmiş görüntüyü genişletme işlemi yapar. Bu işlem, beyaz alanları genişleterek siyah alanları doldurur.

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) ve cnts = imutils.grab_contours(cnts):

    Eşiklenmiş ve genişletilmiş görüntü üzerinde konturları bulur.
    imutils.grab_contours kullanılarak OpenCV 2 ve 3 uyumluluğu sağlanır.

for c in cnts::

    Bulunan konturlar üzerinde döngüye girer.

if cv2.contourArea(c) < conf["min_area"]:

    Eğer bir konturun alanı, yapılandırma dosyasındaki "min_area" değerinden küçükse, bu konturu görmezden gelir.

(x, y, w, h) = cv2.boundingRect(c) ve cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2):

    Konturun sınırlayıcı kutusunu hesaplar ve bu kutuyu çerçeve üzerine çizer.

text = "Occupied":

    Kontur bulunduğunda, oda durumu metni "Occupied" olarak güncellenir.

ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p") ve cv2.putText(...):

    Zaman damgasını (timestamp) belirli bir formata dönüştürür ve bu zaman damgasını ve oda durumu metnini çerçeve üzerine ekler.
	'''
	# check to see if the room is occupied
	if text == "Occupied":
		# check to see if enough time has passed between uploads
		if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
			# increment the motion counter
			motionCounter += 1
			# check to see if the number of frames with consistent motion is
			# high enough
			if motionCounter >= conf["min_motion_frames"]:
				# check to see if dropbox sohuld be used
				if conf["use_dropbox"]:
					# write the image to temporary file
					t = TempImage()
					cv2.imwrite(t.path, frame)
					# upload the image to Dropbox and cleanup the tempory image
					print("[UPLOAD] {}".format(ts))
					path = "/{base_path}/{timestamp}.jpg".format(
					    base_path=conf["dropbox_base_path"], timestamp=ts)
					client.files_upload(open(t.path, "rb").read(), path)
					t.cleanup()
				# update the last uploaded timestamp and reset the motion
				# counter
				lastUploaded = timestamp
				motionCounter = 0
	# otherwise, the room is not occupied
	else:
		motionCounter = 0
	'''
	    if text == "Occupied"::
        Eğer oda durumu "Occupied" ise, aşağıdaki işlemleri gerçekleştirir.

    if (timestamp - lastUploaded).seconds >= conf["min_upload_seconds"]:
        Yeterli zamanın geçip geçmediğini kontrol eder. conf["min_upload_seconds"] parametresi, iki yükleme arasında geçmesi gereken minimum süreyi belirtir.

    motionCounter += 1:
        Hareket sayacını bir artırır.

    if motionCounter >= conf["min_motion_frames"]:
        Hareket sayacının, belirtilen minimum hareket kare sayısına (conf["min_motion_frames"]) eşit veya büyük olup olmadığını kontrol eder.

    if conf["use_dropbox"]:
        Dropbox kullanılacaksa, aşağıdaki işlemleri gerçekleştirir.

    t = TempImage(), cv2.imwrite(t.path, frame), print("[UPLOAD] {}".format(ts)), path = "/{base_path}/{timestamp}.jpg".format(...), ve client.files_upload(...):
        Geçici bir görüntü oluşturur (TempImage sınıfı).
        Oluşturulan geçici görüntüye çerçeveyi yazarak geçici dosyayı oluşturur.
        Dropbox'a yükleme yapar ve yükleme başarılı olursa bir başarı mesajını ekrana yazdırır.

    t.cleanup():
        Geçici dosyayı temizler (silinir).

    lastUploaded = timestamp ve motionCounter = 0:
        Son yüklenen zaman damgasını günceller ve hareket sayacını sıfırlar.

    else::
        Eğer oda durumu "Occupied" değilse, hareket sayacını sıfırlar.
	'''
		# check to see if the frames should be displayed to screen
	if conf["show_video"]: # Eğer yapılandırma dosyasındaki "show_video" değeri True ise, aşağıdaki işlemleri gerçekleştirir.
		# display the security feed
		cv2.imshow("Security Feed", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key is pressed, break from the lop
		if key == ord("q"):
			break
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)