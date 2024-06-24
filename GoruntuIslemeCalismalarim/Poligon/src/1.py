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
kernel = np.ones((3,3), np.uint8)
morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel, iterations=2)

#####################################

# Çember tespiti
circles = cv2.HoughCircles(morph_img, cv2.HOUGH_GRADIENT, 1, img.shape[0]/8, param1=300, param2=150, minRadius=20, maxRadius=100)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 3)

# Sonuçları göster
cv2.imshow("Thresh", thresh)
cv2.imshow("Morfolojik", morph_img)
cv2.imshow("Orijinal", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
