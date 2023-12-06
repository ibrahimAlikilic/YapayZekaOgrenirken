'''
Yalnızca siyah arkaplanlı şekilleri tanır
2 veya daha fazla şekil çakışırsa hepsi tek bir nesne olarak ele alınmalıdır
Siyah şekillerin her birinin etrafındaki konturları tespit edin ve çizin
Siyah şekillerin sayısını sayın
'''
import numpy as np
import argparse
import imutils
import cv2
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image file")
args = vars(ap.parse_args())
# load the image
image = cv2.imread(args["image"])
'''
Now that we have loaded our image off disk, we can move on to detecting the black shapes in the image.

Our Goal: Detect the black shapes in the image.

Detecting these black shapes is actually very easy using the cv2.inRange function:
'''
# find all the 'black' shapes in the image
lower = np.array([0, 0, 0])
upper = np.array([15, 15, 15]) # bunu unutma : And our upper boundary consists of a very dark shade of gray, this time specifying 15 for each of the channels.
shapeMask = cv2.inRange(image, lower, upper)
'''
The next step is to detect the contours in the shapeMask . This is also very straightforward:
'''
# find the contours in the mask
cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("I found {} black shapes".format(len(cnts)))
cv2.imshow("Mask", shapeMask)
# loop over the contours
for c in cnts:
	# draw the contour and show it
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)