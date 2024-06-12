import cv2
import numpy as np

# Resmi yükle
img = cv2.imread('input/2.png')
if img is None:
    raise FileNotFoundError("Görüntü dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#####################################
# Gürültü azaltma

# Blur
img_blurred = cv2.GaussianBlur(img_gray, (15, 15), 0)

# Threshold
thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Morfolojik işlemler
kernel = np.ones((3, 3), np.uint8)
morph_img1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
morph_img1 = cv2.morphologyEx(morph_img1, cv2.MORPH_OPEN, kernel, iterations=3)

morph_img2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
morph_img2 = cv2.morphologyEx(morph_img2, cv2.MORPH_OPEN, kernel, iterations=3)

morph_img3 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
morph_img3 = cv2.morphologyEx(morph_img3, cv2.MORPH_OPEN, kernel, iterations=2)

#####################################

# Çember tespiti
# Tümünü tespit edip ardından noktaları tespit edip hangi aralıkta olduğuna bakacağım .
circles1 = cv2.HoughCircles(morph_img1, cv2.HOUGH_GRADIENT, 1, minDist=50, param1=50, param2=30, minRadius=37, maxRadius=300)
if circles1 is not None:
    circles1 = np.uint16(np.around(circles1))
    for i in circles1[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 3)

circles2 = cv2.HoughCircles(morph_img2, cv2.HOUGH_GRADIENT, 1, minDist=50, param1=50, param2=30, minRadius=37, maxRadius=300)
if circles2 is not None:
    circles2 = np.uint16(np.around(circles2))
    for i in circles2[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 3)

circles3 = cv2.HoughCircles(morph_img3, cv2.HOUGH_GRADIENT, 1, minDist=50, param1=50, param2=30, minRadius=250, maxRadius=300)
if circles3 is not None:
    circles3 = np.uint16(np.around(circles3))
    for i in circles3[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 3)


# tamamını tespit ettikten sonra tespit edilen noktaların merkeze olan uzaklığı ile çemberlerin yarıçaplarını karşılaştırarak puanlamayı yaparsın.

morph_img=morph_img1+morph_img2+morph_img3

# Sonuçları göster
cv2.imshow("Thresh", thresh)
cv2.imshow("Morfolojik", morph_img)
cv2.imshow("Orijinal", img)
print("circles1",circles1)
print("circles2",circles2)
cv2.waitKey(0)
cv2.destroyAllWindows()
