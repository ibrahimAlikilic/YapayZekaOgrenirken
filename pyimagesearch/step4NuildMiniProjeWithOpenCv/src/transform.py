# import the necessary packages
import numpy as np
import cv2
def order_points(pts):
	# initialzie a list of coordinates that will be ordered (sipariş edilecek koordinatların bir listesini başlatın)
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas (sol üst nokta en küçük toplamı alacaktır, oysa)
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

'''
İhtiyacımız olan paketleri içe aktararak başlayacağız: Sayısal işlem için NumPy cv2 OpenCV bağlamalarımız için.

Daha sonra şunu tanımlayalım:order_points. Hat üzerinde çalışır . Bu fonksiyon tek bir argüman alır,
pts dikdörtgenin her noktasının (x, y) koordinatlarını belirten dört noktadan oluşan bir listedir .

Dikdörtgendeki noktaların tutarlı bir sıralamasına sahip olmamız kesinlikle çok önemlidir . Gerçek sıralamanın kendisi, uygulama boyunca tutarlı olduğu sürece keyfi olabilir .

Kişisel olarak puanlarımı sol üst, sağ üst, sağ alt ve sol alt şeklinde belirtmeyi seviyorum.

10. Satırdaki sıralı dört noktaya hafıza ayırarak başlayacağız .

Daha sonra, x + y toplamının en küçük olduğu sol üst noktayı ve x + y toplamının en büyük olduğu sağ alt noktayı bulacağız . Bu , 14-16. Satırlarda ele alınır .

Tabii şimdi sağ üst ve sol alt noktaları bulmamız gerekecek. Burada noktalar arasındaki farkı (yani x – y ) aşağıdakileri kullanarak alacağız: np.diff . Hat üzerinde çalışır . line21

En küçük farka sahip koordinatlar sağ üst noktalar, en büyük farka sahip koordinatlar ise sol alt noktalar olacaktır ( 22. ve 23. satırlar ).

Son olarak sıralı fonksiyonlarımızı 26. Satırdaki çağıran fonksiyona geri döndürüyoruz .
'''

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them (Noktaların tutarlı bir sırasını elde edin ve bunları paketinden çıkarın)
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

'''
We start off by defining the four_point_transform function on Line 28, which requires two arguments: image and pts .

The image variable is the image we want to apply the perspective transform to. And the pts list is the list of four points that contain the ROI of the image we want to transform.

We make a call to our order_points function on Line 31, which places our pts variable in a consistent order. We then unpack these coordinates on Line 32 for convenience.

Now we need to determine the dimensions of our new warped image.

We determine the width of the new image on Lines 37-39, where the width is the largest distance between the bottom-right and bottom-left x-coordinates or the top-right and top-left x-coordinates.

In a similar fashion, we determine the height of the new image on Lines 44-46, where the height is the maximum distance between the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates.

Note: Big thanks to Tom Lowell who emailed in and made sure I fixed the width and height calculation!

So here’s the part where you really need to pay attention.

Remember how I said that we are trying to obtain a top-down, “birds eye view” of the ROI in the original image? And remember how I said that a consistent ordering of the four points representing the ROI is crucial?

On Lines 53-57 you can see why. Here, we define 4 points representing our “top-down” view of the image. The first entry in the list is (0, 0) indicating the top-left corner. The second entry is (maxWidth - 1, 0) which corresponds to the top-right corner. Then we have (maxWidth - 1, maxHeight - 1) which is the bottom-right corner. Finally, we have (0, maxHeight - 1) which is the bottom-left corner.

The takeaway here is that these points are defined in a consistent ordering representation — and will allow us to obtain the top-down view of the image.

To actually obtain the top-down, “birds eye view” of the image we’ll utilize the cv2.getPerspectiveTransform function on Line 60. This function requires two arguments, rect , which is the list of 4 ROI points in the original image, and dst , which is our list of transformed points. The cv2.getPerspectiveTransform function returns M , which is the actual transformation matrix.

We apply the transformation matrix on Line 61 using the cv2.warpPerspective function. We pass in the image , our transform matrix M , along with the width and height of our output image.

The output of cv2.warpPerspective is our warped image, which is our top-down view.

We return this top-down view on Line 64 to the calling function.
'''