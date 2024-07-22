import cv2
import numpy as np
import math

# Resmi yükle
image_path = 'input/2.png'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("Görüntü dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")

#########################################################

## Şimdi merkez koordinatları tespit edeceğiz
# Resmin boyutlarını al
height, width = img.shape[:2]

# Merkez koordinatlarını hesapla
center_x = width // 2
center_y = height // 2
center_x=296
print("merkez koordinatları (x,y) : ",center_x," ",center_y) # 311,296 geldi ama fotoğrafa dikkatli baktığımız zaman poligonun tam olarak ortalanmadığı belli ve biz de merkeze olan uzaklıktan gideceğimizden x= 296 kabul edeceğim

#########################################################

def cemberlerinTespiti(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

    #########################################################

    # Çember tespiti ve çizim
    def contourCizim(circles, b, g, r):
        for i in range(len(circles)):
            if hierarchy[0][i][3]==-1:
               cv2.drawContours(img,circles,i,255,1)
    contours, hierarchy = cv2.findContours(result_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contourCizim(contours, 0, 255, 0)
    print("circles1 : ")
    print(contours) # ilk elemna x 2. eleman y

    
    ##########################################

    # Kontur noktalarının merkezden olan uzaklıklarını hesapla
    distance_dict = {}
    for contour in contours:
        for point in contour:
            x, y = point[0]
            distance = int(math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2))  # Mesafeyi tam sayı olarak hesapla
            # Sadece mesafe daha önce hesaplanmadıysa yazdır
            if distance in distance_dict:
                distance_dict[distance] += 1
                '''
                * Bu satır, distance_dict sözlüğünde distance anahtarına karşılık gelen değeri 1 artırır.
                * Bu, distance değeri daha önce hesaplanmış ve sözlüğe eklenmişse, bu mesafeyi tekrar bulduğumuzu ifade eder ve sayacını artırır.
                '''
            else:
                distance_dict[distance] = 1 # Bu satır, distance_dict sözlüğüne yeni bir distance anahtarı ekler ve değerini 1 olarak ayarlar.

    # Sözlüğü küçükten büyüğe sırala
    sorted_distances = dict(sorted(distance_dict.items()))
    contours_sayisi=0
    for distance, count in sorted_distances.items():
        if count<10:
            pass
        else:
            contours_sayisi+=1
            print(f"Uzaklık: {distance}, Sayı: {count}")
    # aşağıdaki döngü çevrenin noktalardan oluşmasından yola çıkarak oluşturulup r nin bulunması sağlanmıştır input kısmında "ispat.png" adında nasıl hesaplandığı gösterilmiştir. 
    toplamCount=0
    for distance, count in sorted_distances.items():
        if count<10:
            pass
        else:
            toplamCount+=count
    pi=math.pi
    r=toplamCount/(pi*30)
    # Sonuçları göster
    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Morfolojik", morph_img)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Combined Mask", combined_mask)
    # cv2.imshow("Result Mask", result_mask)
    return r

#########################################################

def isabetMerkezKoordinat(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    # Dilatasyon işlemi ekleyin
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(threshold, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #########################################################

    max_contour_area = 500  


    contour_centers = [] # Kontur merkez koordinatlarını tutmak için  dizi

    # Bulunan contourların her birini dikdörtgen içerisine al
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Merkez koordinatlarını hesapla
            center_x = x + w // 2
            center_y = y + h // 2
            contour_centers.append((center_x, center_y))  # Merkez koordinatlarını diziye ekle
            # Sonuçları göster
            # cv2.imshow('Blurred Area', blurred)
            # cv2.imshow('Threshold Area', threshold)
            # cv2.imshow('Dilated Area', dilated)
            cv2.imshow('Detected Area', img)

    return contour_centers

#########################################################

def MerkezeOlanUzaklikFonksiyonu(contour_centers):
    for center in contour_centers:
        farklarinKaresi=((center[0]-center_x)**2) + ((center[1]-center_y)**2)
        merkezUzaklik=math.sqrt(farklarinKaresi)
        return merkezUzaklik

#########################################################

def PanHesapla(r,contour_centers):
    if(contour_centers<=r):
        return 1
    elif(contour_centers<=r*2):
        return 2 
    elif(contour_centers<=r*3):
        return 3
    elif(contour_centers<=r*4):
        return 4 
    elif(contour_centers<=r*5):
        return 5 
    else:
        return 0
    
#########################################################

# Çemberleri tespit et
img_cemberlerinTespiti=img.copy()
r=cemberlerinTespiti(img_cemberlerinTespiti)
print(f"r = {r}")

#########################################################

# Hedefleri tespit et
img_cisabetMerkezKoordinat=img.copy()
contour_centers=isabetMerkezKoordinat(img_cisabetMerkezKoordinat) # (x,y) olarak değer dönüyor
'''
# Kontur merkez koordinatlarını yazdır
print("Kontur Merkez Koordinatları:")
for center in contour_centers:
    print(center)
'''

#########################################################

# Merkeze olanuzaklık hesapla
merkezUzaklik=MerkezeOlanUzaklikFonksiyonu(contour_centers)

#########################################################

# Puan hesapla
puan=PanHesapla(r,contour_centers)

# Sonuçları göster
cv2.imshow("Orijinal", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
