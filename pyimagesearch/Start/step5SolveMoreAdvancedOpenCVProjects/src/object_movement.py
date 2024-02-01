from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")
args = vars(ap.parse_args())
'''
Bu kod satırları, Python'un argparse modülünü kullanarak komut satırı argümanlarını analiz etmek için bir araç oluşturur ve ardından bu argümanları işler. İşte her bir satırın ayrıntılı açıklamaları:

    ap = argparse.ArgumentParser(): argparse modülünden bir ArgümanParser nesnesi oluşturuluyor. Bu nesne, komut satırı argümanlarını işlemek için bir araç sağlar.

    ap.add_argument("-v", "--video", help="path to the (optional) video file"): -v veya --video olarak belirtilen bir komut satırı argümanı ekleniyor. 
    Bu argüman, opsiyonel bir video dosyasının yolu olarak kullanılabilir. help parametresi, kullanıcıya bu argümanın ne işe yaradığını anlatan bir açıklama sağlar.

    ap.add_argument("-b", "--buffer", type=int, default=32, help="max buffer size"): -b veya --buffer olarak belirtilen bir diğer komut satırı argümanı ekleniyor. 
    Bu argüman, bir tam sayı tipinde ve varsayılan değeri 32 olan bir tampon boyutunu belirtir.
        type=int: Bu, argümanın değerini bir tamsayıya dönüştürmek için kullanılan bir tip belirten bir parametredir.
        default=32: Bu, eğer bu argüman belirtilmezse kullanılacak varsayılan değeri belirtir.
        help="max buffer size": Bu, kullanıcıya argümanın amacını açıklayan bir açıklama sağlar.

    args = vars(ap.parse_args()): Komut satırından alınan argümanlar, parse_args() fonksiyonu kullanılarak ayrıştırılır ve args adlı bir sözlükte saklanır. Bu sözlük, argüman isimleriyle (örneğin, "video" ve "buffer") ilişkilendirilmiş değerleri içerir.
'''

# define the lower and upper boundaries of the "green"
# ball in the HSV color space
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY) = (0, 0)
direction = ""
'''
Bu satırlar, topun hareketini izlemek için kullanılan değişkenleri başlatır.

    pts: deque veri yapısını kullanarak, topun izini sürmek için kullanılan bir liste. maxlen parametresi, listenin maksimum uzunluğunu belirler. Bu, geçmişteki belirli bir sayıdaki konumu saklamak için kullanılır.
    counter: Topun kaç kez takip edildiğini sayan bir sayıcı.
    (dX, dY): X ve Y koordinatlarındaki değişimleri temsil eden iki değişken.
    direction: Topun hareket yönünü belirten bir dize.
'''
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
'''
Bu bölümde, kullanıcı bir video dosyası belirtmemişse, bir web kamerasına erişim sağlamak üzere VideoStream sınıfını kullanır. 
Eğer bir video dosyası belirtilmişse, cv2.VideoCapture kullanılarak belirtilen video dosyasına erişim sağlanır.
'''
# allow the camera or video file to warm up
time.sleep(2.0)
'''
Kamera veya video dosyasının başlamasını beklemek için bir bekleme süresi eklenir. 
Bu, cihazın başlaması ve video akışının düzgün çalışması için bir süre beklemeyi sağlar. time.sleep(2.0), 2 saniye boyunca bekleyeceği anlamına gelir.
'''
# keep looping
while True:
	# grab the current frame
	frame = vs.read()
	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	'''
	Bu bölümde, yeşil renk aralığına uygun bir maske oluşturulur (cv2.inRange). 
	Ardından, bu maskeyi birkaç kez erozyon (cv2.erode) ve dilasyon (cv2.dilate) işlemlerinden geçirilir. 
	Bu, maskeyi temizlemek ve küçük parçacıkları kaldırmak için kullanılır.
	'''
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	'''
	Burada, maske üzerindeki konturları (cv2.findContours) buluruz. imutils.grab_contours ile konturları uygun şekilde alırız. 
	center değişkeni, topun merkez koordinatlarını saklamak için kullanılır.
	'''
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		'''
		((x, y), radius) = cv2.minEnclosingCircle(c): Seçilen konturun etrafını saran en küçük çemberi ve çemberin yarıçapını bulur.
		'''
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
			pts.appendleft(center)
			'''
			pts.appendleft(center): pts listesine ağırlık merkezi koordinatlarını ekler. 
			Bu, topun hareketini izlemek için kullanılır ve önceki konumları takip etmek amacıyla bir deque kullanıldığından, listeye eleman eklenirken eski elemanlar 
			otomatik olarak silinir.
			'''
			# loop over the set of tracked points
	for i in np.arange(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None:
			continue
		'''
		Bu bölümde, pts listesindeki takip edilen noktalar üzerinde bir döngü başlatılır. Her iterasyonda, iki ardışık takip edilen nokta kontrol edilir. 
		Eğer herhangi biri None (boş) ise, bu noktalar göz ardı edilir ve döngüye devam edilir.
		'''
		# check to see if enough points have been accumulated in
		# the buffer
		if counter >= 10 and i == 1 and pts[-10] is not None:
			# compute the difference between the x and y
			# coordinates and re-initialize the direction
			# text variables
			dX = pts[-10][0] - pts[i][0]
			dY = pts[-10][1] - pts[i][1]
			(dirX, dirY) = ("", "")
			'''
			Bu bölümde, yeterli sayıda noktanın birikip birikmediği kontrol edilir. 
			Eğer counter (sayac) 10 veya daha fazlaysa ve şu anki noktanın indeksi 1 ise ve 10 önceki nokta boş değilse, hareketin yönünü hesaplamak için 
			farklar alınır.

            dX = pts[-10][0] - pts[i][0]: X koordinatları arasındaki fark.
            dY = pts[-10][1] - pts[i][1]: Y koordinatları arasındaki fark.
            (dirX, dirY) = ("", ""): X ve Y yönlendirmelerini temsil eden değişkenler.
	        '''
			# ensure there is significant movement in the
			# x-direction
			if np.abs(dX) > 20:
				dirX = "East" if np.sign(dX) == 1 else "West"
			# ensure there is significant movement in the
			# y-direction
			if np.abs(dY) > 20:
				dirY = "North" if np.sign(dY) == 1 else "South"
				'''
				Burada, X ve Y yönlendirmelerinin belirlenmesi için farklar kontrol edilir. 
				Eğer X veya Y yönlendirmeleri 20 birimden büyükse, hareket anlamlı kabul edilir ve bu durumda dirX ve/veya dirY değişkenleri güncellenir.
				'''
			# handle when both directions are non-empty
			if dirX != "" and dirY != "":
				direction = "{}-{}".format(dirY, dirX)
			# otherwise, only one direction is non-empty
			else:
				direction = dirX if dirX != "" else dirY
				'''
				Eğer hem dirX hem de dirY değişkenleri doluysa (hareket hem X hem de Y yönlendirmelerinde varsa), bu iki yön birleştirilir ve direction değişkenine atanır. 
				Aksi halde, sadece bir yönde hareket varsa, bu yönde olan değişken direction değişkenine atanır.
				'''
					# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
		'''
		Bu kısım, izlenen noktalar arasında bir bağlantı çizgisi çizer. Bağlantı çizgisinin kalınlığı, izlenen nokta sayısına bağlı olarak ayarlanır. 
		Daha fazla izlenen nokta varsa, çizgi daha ince olur. cv2.line ile çizgi çizilir.
		'''
	# show the movement deltas and the direction of movement on
	# the frame
	cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (0, 0, 255), 3)
	cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
		0.35, (0, 0, 255), 1)
	'''
	Bu kısımda, ekrana hareket yönü ve X, Y koordinatlarındaki değişim miktarları yazdırılır. cv2.putText fonksiyonu kullanılarak metin eklenir.
	'''
	# show the frame to our screen and increment the frame counter
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	counter += 1
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
	'''
	Bu kısım, ekrandaki çerçeveyi gösterir (cv2.imshow). Ardından, kullanıcının bir tuşa basmasını bekler ve eğer basılan tuş 'q' ise döngüden çıkar. 
	Bu, programın sonlandırılmasını sağlar.
	'''
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()