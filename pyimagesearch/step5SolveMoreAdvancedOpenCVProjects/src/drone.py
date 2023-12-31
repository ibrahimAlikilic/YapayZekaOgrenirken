import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())
# load the video
camera = cv2.VideoCapture(args["video"])
# keep looping
while True:
	# grab the current frame and initialize the status text
	(grabbed, frame) = camera.read()
	status = "No Targets"
	# check to see if we have reached the end of the
	# video
	if not grabbed:
		break
	# convert the frame to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 0)
	edged = cv2.Canny(blurred, 50, 150)
	# find contours in the edge map
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	
	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)
		# ensure that the approximated contour is "roughly" rectangular
		if len(approx) >= 4 and len(approx) <= 6:
			# compute the bounding box of the approximated contour and
			# use the bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			aspectRatio = w / float(h)
			# compute the solidity of the original contour
			area = cv2.contourArea(c)
			hullArea = cv2.contourArea(cv2.convexHull(c))
			solidity = area / float(hullArea)
			# compute whether or not the width and height, solidity, and
			# aspect ratio of the contour falls within appropriate bounds
			keepDims = w > 25 and h > 25
			keepSolidity = solidity > 0.9
			keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2
			# ensure that the contour passes all our tests
			if keepDims and keepSolidity and keepAspectRatio:
				# draw an outline around the target and update the status
				# text
				cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
				status = "Target(s) Acquired"
				# compute the center of the contour region and draw the
				# crosshairs
				M = cv2.moments(approx)
				(cX, cY) = (int(M["m10"] // M["m00"]), int(M["m01"] // M["m00"]))
				(startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15)))
				(startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
				cv2.line(frame, (startX, cY), (endX, cY), (0, 0, 255), 3)
				cv2.line(frame, (cX, startY), (cX, endY), (0, 0, 255), 3)
				'''
				    for c in cnts:: Konturlar üzerinde döngü başlar.

    peri = cv2.arcLength(c, True): Konturun çevresini hesaplar.

    approx = cv2.approxPolyDP(c, 0.01 * peri, True): Konturu düzleştirir (approximates) ve daha düzgün bir şekle getirir.

    if len(approx) >= 4 and len(approx) <= 6:: Eğer düzeltilmiş kontur 4 ile 6 arasında köşe içeriyorsa devam eder.

    (x, y, w, h) = cv2.boundingRect(approx): Düzleştirilmiş konturun çevresini saran dikdörtgenin koordinatlarını ve boyutlarını hesaplar.

    aspectRatio = w / float(h): Dikdörtgenin en-boy oranını hesaplar.

    area = cv2.contourArea(c): Orijinal konturun alanını hesaplar.

    hullArea = cv2.contourArea(cv2.convexHull(c)): Orijinal konturun dış kabuğunun alanını hesaplar.

    solidity = area / float(hullArea): Orijinal konturun sağlamlığını (solidity) hesaplar.

    keepDims = w > 25 and h > 25: Genişlik ve yükseklik belirli bir eşiği geçiyorsa devam eder.

    keepSolidity = solidity > 0.9: Sağlamlık belirli bir eşiği geçiyorsa devam eder.

    keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2: En-boy oranı belirli bir aralıkta ise devam eder.

    if keepDims and keepSolidity and keepAspectRatio:: Eğer tüm testler başarılıysa devam eder.

    cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4): Hedefi çevreleyen bir çizgi çizer.

    status = "Target(s) Acquired": Hedef tespit edildiğinde durumu günceller.

    M = cv2.moments(approx): Konturun momentlerini hesaplar.

    (cX, cY) = (int(M["m10"] // M["m00"]), int(M["m01"] // M["m00"])): Konturun merkezini hesaplar.

    (startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0.15))): Çizilen çapraz işaretin başlangıç ve bitiş koordinatlarını hesaplar.

    (startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15))): Çizilen çapraz işaretin başlangıç ve bitiş koordinatlarını hesaplar.

    cv2.line(frame, (startX, cY), (endX, cY), (0, 0, 255), 3): Çapraz işareti çizer (yatay).

    cv2.line(frame, (cX, startY), (cX, endY), (0, 0, 255), 3): Çapraz işareti çizer (dikey).

Bu kısım, belirli kriterlere uyan konturları bulur ve bunları çizim ve diğer işlemlerle işler.
'''
	# draw the status text on the frame
	cv2.putText(frame, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		(0, 0, 255), 2)
	# show the frame and record if a key is pressed
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()