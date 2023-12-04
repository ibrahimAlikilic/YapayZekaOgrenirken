# import the necessary packages
from imutils import build_montages # indirmeler kilitliydi teorik devam
from imutils import paths # indirmeler kilitliydi teorik devam
import argparse
import random
import cv2
# construct the argument parse and parse the arguments (argümanı oluşturun argümanları ayrıştırın ve ayrıştırın)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
ap.add_argument("-s", "--sample", type=int, default=21,
	help="# of images to sample")
args = vars(ap.parse_args()) # daha iyi anlamak için bulduğum site : https://medium.com/@celikemirhan/python-argument-parser-kullanimi-50511bd6f609
'''
--images : The path to your directory containing the images you want to build a montage out of.
--samples : An optional command line argument that specifies the number of images to sample (we default this value to 21 total images).
'''
# grab the paths to the images, then randomly select a sample of
# them
imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths) # Bu fonksiyon, bir listede yada tupleda yer alan öğeleri rastgele getirir.
imagePaths = imagePaths[:args["sample"]]

# initialize the list of images
images = []
# loop over the list of image paths
for imagePath in imagePaths:
	# load the image and update the list of images
	image = cv2.imread(imagePath)
	images.append(image)
# construct the montages for the images
montages = build_montages(images, (128, 196), (7, 3))
'''
image_list : This parameter is a list of images loaded via OpenCV. In our case, we supply the images list built on Lines 26-29.
image_shape : A tuple containing the width and height of each image in the montage. Here we indicate that all images in the montage will be resized to 129 x 196. Resizing every image in the montage to a fixed size is a requirement so we can properly allocate memory in the resulting NumPy array. Note: Empty space in the montage will be filled with black pixels.
montage_shape : A second tuple, this one specifying the number of columns and rows in the montage. Here we indicate that our montage will have 7 columns (7 images wide) and 3 rows (3 images tall).
'''

# loop over the montages and display each of them
for montage in montages:
	cv2.imshow("Montage", montage)
	cv2.waitKey(0)
 