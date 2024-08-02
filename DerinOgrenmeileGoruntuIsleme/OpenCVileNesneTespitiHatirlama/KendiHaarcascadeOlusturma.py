"""
1) veri seti : n,p
2) cascade programı indir
3) cascade oluştur
4) cascade kullanarak
"""

import cv2
import os

# Kamera çözünürlüğü ve parlaklık seviyesini almak için fonksiyon
def Cozunurluk(cap):
    width = 0
    height = 0
    brightness = 0
    if not cap.isOpened():
        print("Kamera açılamadı!")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        print(f"Kamera Çözünürlüğü: {width} x {height}")
        print(f"Kamera Parlaklık Seviyesi: {brightness}")
    return width, height, brightness

##########################################


# Kamera başlatma ve ayarlarını yapma fonksiyonu
def CapSet():
    cap = cv2.VideoCapture(0)
    width, height, brightness = Cozunurluk(cap)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    return cap

##########################################

# Yeni bir klasör oluşturmak için method
def Dosyalama(klasor_adi):
    tam_yol = os.path.join(path, str(klasor_adi))
    if not os.path.exists(tam_yol):
        os.makedirs(tam_yol)

##########################################

# NegatifValue ve PositiveValue methodunu tek bir method olarak oluşturup parametre alarak istediğime kaydettirebilirdim ama bu dersi yeni izliyorum ve orada hiç böyle yapmamış ben kendim insiyatif alarak methodlar oluşturdum ileride ne yapacağını bilmediğimden ayrı ayrı yaptım

# NegatifValue method
def NegatifValue():
    global count, countSaveN
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (img_width, img_height))
            if count % 5 == 0:
                cv2.imwrite(os.path.join(path, str('n'), f"{countSaveN}_.png"), frame)
                countSaveN += 1
                print(countSaveN)
            count += 1
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        elif not ret:
            print("Okuyamadım")

# PositiveValue method
def PositiveValue():
    global count, countSaveP
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (img_width, img_height))
            if count % 5 == 0:
                cv2.imwrite(os.path.join(path, str('p'), f"{countSaveP}_.png"), frame)
                countSaveP += 1
                print(countSaveP)
            count += 1
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        elif not ret:
            print("Okuyamadım")

# Ana değişkenler
global cap 
cap = CapSet()
global path
path = "output"
img_width = 180
img_height = 120
count = 0
countSaveN = 0
countSaveP = 0

# Ana döngü

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            # Dosyalama işlemi başlatılıyor
            Dosyalama('n')
            NegatifValue() # veri kaydetme işlemi
        elif key == ord('p'):
            Dosyalama('p')
            PositiveValue() # veri kaydetme işlemi
        elif key == ord('q'):
            break
    elif not ret:
        print("Okuyamadım")

cap.release()
cv2.destroyAllWindows()
