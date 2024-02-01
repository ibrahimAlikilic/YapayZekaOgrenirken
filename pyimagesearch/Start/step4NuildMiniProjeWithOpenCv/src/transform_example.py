from transform import four_point_transform # aynı dizinde diye bu şekilde import ettik farklı olsalardı
'''
import sys
sys.path.append('/path/to/transform/directory')

from transform import four_point_transform

gibi olacaktı.
'''
import numpy as np
import argparse
import cv2
# construct the argument parse and parse the arguments (argümanı oluşturun argümanları ayrıştırın ve ayrıştırın)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "\input")
ap.add_argument("-c", "--coords",
	help = "comma seperated list of source points")
args = vars(ap.parse_args())
# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them
image = cv2.imread(args["image"])
pts = np.array(eval(args["coords"]), dtype = "float32")
# apply the four point tranform to obtain a "birds eye view" of ("kuş bakışı" bir görünüm elde etmek için dört nokta dönüşümünü uygulayın)
# the image
warped = four_point_transform(image, pts)
# show the original and warped images
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
'''
The first thing we’ll do is import our four_point_transform function on Line 2. I decided put it in the pyimagesearch sub-module for organizational purposes.

We’ll then use NumPy for the array functionality, argparse for parsing command line arguments, and cv2 for OpenCV bindings.

We parse our command line arguments on Lines 8-12. We’ll use two switches, --image , which is the image that we want to apply the transform to, and --coords , which is the list of 4 points representing the region of the image we want to obtain a top-down, “birds eye view” of.

We then load the image on Line 19 and convert the points to a NumPy array on Line 20.
'''