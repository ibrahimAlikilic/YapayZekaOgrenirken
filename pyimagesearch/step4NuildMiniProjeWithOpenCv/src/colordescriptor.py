# import the necessary packages paketin var olduğu sitenin süresi dolmuş varmış gibi yol alacağım
# link : https://pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/


import numpy as np
import cv2
import imutils
class ColorDescriptor:
	def __init__(self, bins):
		# store the number of bins for the 3D histogram
		self.bins = bins
	def describe(self, image):
		# convert the image to the HSV color space and initialize
		# the features used to quantify the image (görüntüyü ölçmek için kullanılan özellikler)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		# grab the dimensions and compute the center of the image (boyutları alın ve görüntünün merkezini hesaplayın)
		(h, w) = image.shape[:2]
		(cX, cY) = (int(w * 0.5), int(h * 0.5))
  
        
        
        # divide the image into four rectangles/segments (top-left,
		# top-right, bottom-right, bottom-left)
		segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
			(0, cX, cY, h)]
		# construct an elliptical mask representing the center of the
		# image
		(axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
		# loop over the segments
		for (startX, endX, startY, endY) in segments:
			# construct a mask for each corner of the image, subtracting
			# the elliptical center from it
			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipMask)
			# extract a color histogram from the image, then update the
			# feature vector
			hist = self.histogram(image, cornerMask)
			features.extend(hist)
		# extract a color histogram from the elliptical region and
		# update the feature vector
		hist = self.histogram(image, ellipMask)
		features.extend(hist)
		# return the feature vector
		return features
'''
İhtiyacımız olan Python paketlerini içe aktararak başlayacağız. Sayısal işlemler için NumPy'yi kullanacağız.cv2 
OpenCV bağlamalarımız için ve imutils OpenCV sürümünü kontrol etmek için.

Daha sonra tanımlarımızı tanımlıyoruz ColorDescriptor 6. Hat'taki sınıf . Bu sınıf, 3D HSV renkli histogramımızı görüntülerimizden çıkarmak için gerekli tüm mantığı kapsayacaktır.

The __init__ method of the ColorDescriptor takes only a single argument, bins , which is the number of bins for our color histogram.

We can then define our describe method on Line 11. This method requires an image , which is the image we want to describe.

Inside of our describe method we’ll convert from the RGB color space (or rather, the BGR color space; OpenCV represents RGB images as NumPy arrays, but in reverse order) to the HSV color space, 
followed by initializing our list of features to quantify and represent our image .

Lines 18 and 19 simply grab the dimensions of the image and compute the center (x, y) coordinates.

So now the hard work starts.

Instead of computing a 3D HSV color histogram for the entire image, let’s instead compute a 3D HSV color histogram for different regions of the image.

Using regions-based histograms rather than global-histograms allows us to simulate locality in a color distribution.
'''

'''
Lines 23 and 24 start by defining the indexes of our top-left, top-right, bottom-right, and bottom-left regions, respectively.

From there, we’ll need to construct an ellipse to represent the center region of the image. We’ll do this by defining an ellipse radius that is 75% of the width and height of the image on Line 28.

We then initialize a blank image (filled with zeros to represent a black background) with the same dimensions of the image we want to describe on Line 29.

Finally, let’s draw the actual ellipse on Line 30 using the cv2.ellipse function. This function requires eight different parameters:

ellipMask : The image we want to draw the ellipse on. We’ll be using a concept of “masks” which I’ll discuss shortly.
(cX, cY) : A 2-tuple representing the center (x, y)-coordinates of the image.
(axesX, axesY) : A 2-tuple representing the length of the axes of the ellipse. In this case, the ellipse will stretch to be 75% of the width and height of the image that we are describing.
 0 : The rotation of the ellipse. In this case, no rotation is required so we supply a value of 0 degrees.
 0 : The starting angle of the ellipse.
360 : The ending angle of the ellipse. Looking at the previous parameter, this indicates that we’ll be drawing an ellipse from 0 to 360 degrees (a full “circle”).
255 : The color of the ellipse. The value of 255 indicates “white”, meaning that our ellipse will be drawn white on a black background.
-1 : The border size of the ellipse. Supplying a positive integer r will draw a border of size r pixels. Supplying a negative value for r will make the ellipse “filled in”.
We then allocate memory for each corner mask on Line 36, draw a white rectangle representing the corner of the image on Line 37, and then subtract the center ellipse from the rectangle on 
Line 38.
'''

def histogram(self, image, mask): # okurken burayı line 53 diye düşün.
		# extract a 3D color histogram from the masked region of the
		# image, using the supplied number of bins per channel
		hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
			[0, 180, 0, 256, 0, 256])
		# normalize the histogram if we are using OpenCV 2.4
		if imutils.is_cv2():
			hist = cv2.normalize(hist).flatten()
		# otherwise handle for OpenCV 3+ (OpenCV 3+ için therwise tutamacı)
		else:
			hist = cv2.normalize(hist, hist).flatten()
		# return the histogram
		return hist

'''
Our histogram method requires two arguments: the first is the image that we want to describe and the second is the mask that represents the region of the image we want to describe.

Calculating the histogram of the masked region of the image is handled on Lines 56 and 57 by making a call to cv2.calcHist using the supplied number of bins from our constructor.

Our color histogram is normalized on Line 61 or 65 (depending on OpenCV version) to obtain scale invariance. This means that if we computed a color histogram for two identical images, except that one was 50% larger than the other, our color histograms would be (nearly) identical. It is very important that you normalize your color histograms so each histogram is represented by the relative percentage counts for a particular bin and not the integer counts for each bin. Again, performing this normalization will ensure that images with similar content but dramatically different dimensions will still be “similar” once we apply our similarity function.

Finally, the normalized, 3D HSV color histogram is returned to the calling function on Line 68.
'''