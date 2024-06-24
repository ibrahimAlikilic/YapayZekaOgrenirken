import cv2
import numpy as np

# Resmi yükle
image_path = 'input/2.png'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Görüntü dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gürültü azaltma
# Blur
img_blurred = cv2.GaussianBlur(img_gray, (15, 15), 0)
# Siyah rengi tespit et
# HSV renk uzayına dönüştür
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])
mask = cv2.inRange(img_hsv, lower_black, upper_black)

# Morfolojik işlemler
kernel = np.ones((3, 3), np.uint8)
morph_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_OPEN, kernel, iterations=3)


# Kontur bulma
contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Çember çizim fonksiyonu
def contourCizim(contours, b, g, r):
    for contour in contours:
        # Minumum çevreleyen çemberi bul
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > 10:  # Minimum yarıçap filtresi
            # Çemberi çiz
            cv2.circle(img, center, radius, (b, g, r), 3)
            # Merkezi noktayı çiz
            cv2.circle(img, center, 2, (0, 0, 255), 3)
            # Merkez ve yarıçap bilgilerini yazdır
            cv2.putText(img, f"Center: ({center[0]}, {center[1]})", (center[0] - 50, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, f"Radius: {radius}", (center[0] - 50, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Çemberin merkezinden çembere bir çizgi çiz
            cv2.line(img, center, (center[0] + radius, center[1]), (255, 255, 255), 2)

# Konturları kullanarak çemberleri çiz
contourCizim(contours, 0, 255, 0)

# Sonuçları göster
cv2.imshow("Mask", mask)
cv2.imshow("Morfolojik", morph_img)
cv2.imshow("Orijinal", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
