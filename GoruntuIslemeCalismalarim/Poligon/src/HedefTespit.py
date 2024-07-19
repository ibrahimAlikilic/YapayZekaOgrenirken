import cv2
import numpy as np

# Görseli oku
image = cv2.imread('input/6.png')

# Görseli gri tonlamalı formata çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Bulanıklaştırma işlemi (Gaussian Blur)
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# Gri renge yakın bölgeleri eşikleyin
_, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# Dilatasyon işlemi ekleyin
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(threshold, kernel, iterations=2)

# Eşiklenmiş ve dilate edilmiş görseldeki konturları bulun
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max_contour_area = 500  

# Bulunan konturlar üzerinde dön ve her birini dikdörtgen içerisine al
for contour in contours:
    area = cv2.contourArea(contour)
    if area < max_contour_area:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Dikdörtgen çiz (yeşil renk, kalınlık 2)

# Sonucu göster
cv2.imshow('Blurred Area', blurred)
cv2.imshow('Threshold Area', threshold)
cv2.imshow('Dilated Area', dilated)
cv2.imshow('Detected Area', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Sonucu kaydetmek isterseniz
cv2.imwrite('/mnt/data/detected_areas.png', image)
