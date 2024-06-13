import cv2
import numpy as np

# Resmi yükle
image_path = 'input/2.png'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Görüntü dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#########################################################33

# Gürültü azaltma

# Blur
img_blurred = cv2.GaussianBlur(img_gray, (15, 15), 0)

# Threshold
ret,thresh=cv2.threshold(img_blurred,200,255,cv2.THRESH_BINARY) 

# Siyah rengi tespit et . Bu sayede daha net çıktı alıyorum . 
# Amacım lekeleri temizleyip ana çemberleri bulmak olduğundan bunun faydalıolabileceğini düşünüp denedim ve oldu 
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])
mask = cv2.inRange(img_hsv, lower_black, upper_black)

# Morfolojik işlemler
kernel = np.ones((3, 3), np.uint8)
morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel, iterations=3)

## Şimdi oluşan çıktılardaki bitwise_or ile daha net olmasını sağlayacağım

# Maskeyi ve morfolojik işlemden geçen görüntüyü birleştir
combined_mask = cv2.bitwise_or(mask, morph_img)

# Sonuç maskesi
result_mask = cv2.bitwise_and(combined_mask, thresh)
#########################################################33
# Çember tespiti ve çizim
def contourCizim(circles, b, g, r):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (b, g, r), 3)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.putText(img, f"Center: ({i[0]}, {i[1]})", (i[0] - 50, i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(img, f"Radius: {i[2]}", (i[0] - 50, i[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.line(img, (i[0], i[1]), (i[0] + i[2], i[1]), (0, 0, 255), 2)

circles1 = cv2.HoughCircles(result_mask, cv2.HOUGH_GRADIENT, 1, minDist=50, param1=50, param2=30, minRadius=250, maxRadius=300)
contourCizim(circles1, 0, 255, 0)

# Sonuçları göster
cv2.imshow("Thresh", thresh)
cv2.imshow("Morfolojik", morph_img)
cv2.imshow("Mask", mask)
cv2.imshow("Combined Mask", combined_mask)
cv2.imshow("Result Mask", result_mask)
cv2.imshow("Orijinal", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
