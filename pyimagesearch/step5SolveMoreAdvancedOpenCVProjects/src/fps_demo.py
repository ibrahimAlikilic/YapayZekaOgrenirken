# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())
'''
Bu kısım, betiğin komut satırından argümanları almasını sağlar. argparse kütüphanesi kullanılarak komut satırı argümanları analiz edilir. İki ana argüman vardır:

-n veya --num-frames: FPS testi için kaç kare üzerinden geçileceği. Varsayılan olarak 100 kare belirlenmiştir.
-d veya --display: Kameradan alınan karelerin ekranda gösterilip gösterilmeyeceği. -1 değeri, karelerin gösterilmemesi anlamına gelir.
Bu argümanlar, ap.parse_args() çağrısı ile analiz edilir ve args adlı bir sözlük içine yerleştirilir.
'''
# grab a pointer to the video stream and initialize the FPS counter
print("[INFO] sampling frames from webcam...")
stream = cv2.VideoCapture(0)
fps = FPS().start()
# loop over some frames
while fps._numFrames < args["num_frames"]:
	# grab the frame from the stream and resize it to have a maximum
	# width of 400 pixels
	(grabbed, frame) = stream.read()
	frame = imutils.resize(frame, width=400)
	
 # check to see if the frame should be displayed to our screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	# update the FPS counter
	fps.update()
'''
Bu kısım, belirtilen sayıda kare üzerinden bir döngü çalıştırır. Her iterasyonda:

stream.read() ile bir sonraki kare alınır. grabbed değişkeni, kare alınıp alınamadığını gösterir.
imutils.resize ile kare, maksimum genişliği 400 piksel olacak şekilde yeniden boyutlandırılır.
'''
'''
Bu kısım, eğer kareler ekrana gösterilecekse (args["display"] > 0), kare ekranda gösterilir. cv2.waitKey ile bir tuşa basılırsa (key = cv2.waitKey(1) & 0xFF), döngüden çıkılır.
'''
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()
'''
Son olarak, kullanılan kaynakları temizlemek için stream.release() ve cv2.destroyAllWindows() kullanılır. stream.release() ile kamera kaynağı serbest bırakılır ve cv2.destroyAllWindows() ile açılmış olan tüm penceler kapatılır.
'''
# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
# loop over some frames...this time using the threaded stream
while fps._numFrames < args["num_frames"]:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	# check to see if the frame should be displayed to our screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	# update the FPS counter
	fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
'''
Bu kısımda, WebcamVideoStream(src=0).start() ile iş parçacıklı bir video akışı oluşturulur. Ardından, FPS().start() ile bir FPS sayacı başlatılır.
'''
'''
Bu kısım, belirtilen sayıda kare üzerinden bir döngü çalıştırır. Her iterasyonda:

vs.read() ile iş parçacıklı video akışından bir kare alınır.
imutils.resize ile kare, maksimum genişliği 400 piksel olacak şekilde yeniden boyutlandırılır.
'''