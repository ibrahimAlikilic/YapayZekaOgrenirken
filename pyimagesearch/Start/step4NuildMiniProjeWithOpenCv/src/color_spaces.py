# import the necessary packages
import argparse
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="adrian.png",
	help="path to input image")
args = vars(ap.parse_args())
# load the original image and show it
image = cv2.imread(args["image"])
cv2.imshow("RGB", image)
# loop over each of the individual channels and display them
for (name, chan) in zip(("B", "G", "R"), cv2.split(image)):
	cv2.imshow(name, chan)
# wait for a keypress, then close all open windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# convert the image to the HSV color space and show it
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
# loop over each of the individual channels and display them
for (name, chan) in zip(("H", "S", "V"), cv2.split(hsv)):
	cv2.imshow(name, chan)
# wait for a keypress, then close all open windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# LAB

import math
red_green = math.sqrt(((255 - 0) ** 2) + ((0 - 255) ** 2) + ((0 - 0) ** 2))
red_purple = math.sqrt(((255 - 128) ** 2) + ((0 - 0) ** 2) + ((0 - 128) ** 2))
red_navy = math.sqrt(((255 - 0) ** 2) + ((0 - 0) ** 2) + ((0 - 128) ** 2))
red_green, red_purple, red_navy
'''
L-channel: The “lightness” of the pixel. This value goes up and down the vertical axis, white to black, with neutral grays at the center of the axis.
a-channel: Originates from the center of the L-channel and defines pure green on one end of the spectrum and pure red on the other.
b-channel: Also originates from the center of the L-channel, but is perpendicular to the a-channel. The b-channel defines pure blue at one of the spectrum and pure yellow at the other.
'''
# convert the image to the L*a*b* color space and show it
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("L*a*b*", lab)
# loop over each of the individual channels and display them
for (name, chan) in zip(("L*", "a*", "b*"), cv2.split(lab)):
	cv2.imshow(name, chan)
# wait for a keypress, then close all open windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
Gri tonlamalı
Tartışacağımız son renk uzayı aslında bir renk uzayı değildir; yalnızca bir RGB görüntüsünün gri tonlamalı temsilidir.

Bir görüntünün gri tonlamalı gösterimi, görüntünün renk bilgisini ortadan kaldırır ve cv2.cvtColor işlevi kullanılarak da yapılabilir:
'''
# show the original and grayscale versions of the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)