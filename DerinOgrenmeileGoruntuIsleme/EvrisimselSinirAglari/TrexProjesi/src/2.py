import keyboard
import uuid
import time
import cv2
import numpy as np
from PIL import ImageGrab

# Ekran koordinatları (770,490 : 961,536)
mon = {"top": 490, "left": 770, "width": 191, "height": 46}

i = 0

def record_screen(record_id, key):
    global i
    i += 1
    print("{} {}".format(key, i))  # key = klavyede bastığımız tuş, i = kaç kez klavyeye bastığımız

    # Ekran görüntüsünü al
    img = ImageGrab.grab(bbox=(mon["left"], mon["top"], mon["left"] + mon["width"], mon["top"] + mon["height"]))
    img_np = np.array(img)
    
    # BGR formatına dönüştür
    frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Görüntüyü kaydet
    cv2.imwrite("./output/{}_{}_{}.png".format(key, record_id, i), frame)

is_exit = False  # Veri toplama işlemini sonlandırmak için bool oluşturduk.

def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey(hotkey="esc", callback=exit)  # esc tuşuna basınca exit fonksiyonunu çağır

record_id = uuid.uuid4()  # UUID4, rastgele sayılar kullanarak oluşturulan bir UUID çeşididir ve genellikle benzersizliği sağlamak için yeterli bir yöntemdir.

while True:
    if is_exit:
        break
    try:
        if keyboard.is_pressed(keyboard.KEY_UP):  # Yukarı tuşuna basılırsa
            record_screen(record_id, "up")  # Ekran görüntüsünü al
            time.sleep(0.1)  # Her komuttan sonra 0.1sn bekliyoruz yoksa çok hızlı oluyor ve yukarı yerine aşağı komutu çıkabiliyor
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id, "down")
            time.sleep(0.1)
        elif keyboard.is_pressed("right"):  # Hiçbir şey yapmaması gerektiği durumda ileri tuşuna basacağım ki yanlışlıkla zıplama gibi durumlar olmasın ve normal zamanda ekranın akması gerektiğini bilsin
            record_screen(record_id, "right")
            time.sleep(0.1)
    except RuntimeError:
        continue
