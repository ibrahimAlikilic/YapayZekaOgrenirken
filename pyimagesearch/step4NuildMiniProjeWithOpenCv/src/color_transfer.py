'''
The algorithm goes like this:

Step 1: Input a source and a target image. The source image contains the color space that you want your target image to mimic. In the figure at the top of this page, the sunset image on the left is my source, the middle image is my target, and the image on the right is the color space of the source applied to the target.
Step 2: Convert both the source and the target image to the L*a*b* color space. The L*a*b* color space models perceptual uniformity, where a small change in an amount of color value should also produce a relatively equal change in color importance. The L*a*b* color space does a substantially better job mimicking how humans interpret color than the standard RGB color space, and as you’ll see, works very well for color transfer.
Step 3: Split the channels for both the source and target.
Step 4: Compute the mean and standard deviation of each of the L*a*b* channels for the source and target images.
Step 5: Subtract the mean of the L*a*b* channels of the target image from target channels.
Step 6: Scale the target channels by the ratio of the standard deviation of the target divided by the standard deviation of the source, multiplied by the target channels.
Step 7: Add in the means of the L*a*b* channels for the source.
Step 8: Clip any values that fall outside the range [0, 255]. (Note: This step is not part of the original paper. I have added it due to how OpenCV handles color space conversions. If you were to implement this algorithm in a different language/library, you would either have to perform the color space conversion yourself, or understand how the library doing the conversion is working).
Step 9: Merge the channels back together.
Step 10: Convert back to the RGB color space from the L*a*b* space.
'''
import numpy as np
import cv2
def color_transfer(source, target):
	# convert the images from the RGB to L*ab* color space, being
	# sure to utilizing the floating point data type (note: OpenCV
	# expects floats to be 32-bit, so use that instead of 64-bit)
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    # compute color statistics for the source and target images # line 12
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source) #?
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target) #?
	# subtract the means from the target image
	(l, a, b) = cv2.split(target)
	l -= lMeanTar
	a -= aMeanTar
	b -= bMeanTar
	# scale by the standard deviations
	l = (lStdTar / lStdSrc) * l
	a = (aStdTar / aStdSrc) * a
	b = (bStdTar / bStdSrc) * b
	# add in the source mean
	l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc
	# clip the pixel intensities to [0, 255] if they fall outside
	# this range
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)
	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
	
	# return the color transferred image
	return transfer
'''
Lines 13 and 14 make calls to the image_stats function, which I’ll discuss in detail in a few paragraphs. But for the time being, know that this function simply computes the mean and standard deviation of the pixel intensities for each of the L*, a*, and b* channels, respectively (Steps 3 and 4).

Now that we have the mean and standard deviation for each of the L*a*b* channels for both the source and target images, we can now perform the color transfer.

On Lines 17-20, we split the target image into the L*, a*, and b* components and subtract their respective means (Step 5).

From there, we perform Step 6 on Lines 23-25 by scaling by the ratio of the target standard deviation, divided by the standard deviation of the source image.

Then, we can apply Step 7, by adding in the mean of the source channels on Lines 28-30.

Step 8 is handled on Lines 34-36 where we clip values that fall outside the range [0, 255] (in the OpenCV implementation of the L*a*b* color space, the values are scaled to the range [0, 255], although that is not part of the original L*a*b* specification).

Finally, we perform Step 9 and Step 10 on Lines 41 and 42 by merging the scaled L*a*b* channels back together, and finally converting back to the original RGB color space.

Lastly, we return the color transferred image on Line 45.
'''
def image_stats(image):
	# compute the mean and standard deviation of each channel
	(l, a, b) = cv2.split(image)
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())
	# return the color statistics
	return (lMean, lStd, aMean, aStd, bMean, bStd)