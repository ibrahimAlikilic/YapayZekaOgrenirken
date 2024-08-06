import keyboard # klavye üzerindeki tuşları kullanarak veri toplamamıza yardımcı olacak kütüphane
import uuid # bu kütüphane sayesinde ekrandan kayıt alabileceğiz
import time
from PIL import Image
from mss import mss
"""
Oyun linki : http://www.trex-game.skipser.com/
"""
# Derin öğrenme modelimin görmesin istediğim frame nin koordinatlarını ( 770,490 : 961,536 ) belirledim şimdi kodlayalım 
mon = {"top":490 , "left":770 , "width":191 , "height":46 }
sct = mss()
"""
mss kütüphanesi, ekran görüntüleri almak için kullanılan bir Python kütüphanesidir.
Bu kütüphane, özellikle performans açısından verimli bir şekilde ekran görüntüleri almanızı sağlar.
mss kütüphanesini kullanarak belirli bir ekran alanının görüntüsünü alabilir ve bu görüntüyü işlemek üzere kullanabilirsiniz.
"""

i=0
def record_screen(record_id,key):
    global i
    i+=1
    print("{} {}".format(key,i)) # key = klavyede bastığımız tuş , i = kaç kez klavyeye bastığımız
    img=sct.grab(mon) # belirlemiş olduğum değerler doğrultusunda ekran görüntüsünü al
    im=Image.frombytes("RGB",img.size,img.rgb)
    im.save("./output/{}_{}_{}.png".format(key,record_id,i))

is_exit=False # veri toplama işlemini sonlandırmak için bool oluşturduk.
def exit():
    global is_exit
    is_exit=True
keyboard.add_hotkey(hotkey="esc",callback=exit) # esc tuşuna basınca exit fonksiyonumu çağır dedik.

recor_id=uuid.uuid4() # UUID4, rastgele sayılar kullanarak oluşturulan bir UUID çeşididir ve genellikle benzersizliği sağlamak için yeterli bir yöntemdir.

while True:
    if is_exit: break
    try:
        if keyboard.is_pressed(keyboard.KEY_UP): # yukarı tuşuna basılırsa
            record_screen(recor_id,"up") # ekran görüntüsünü al
            time.sleep(0.1)# her komuttan sonra 0.1sn bekliyoruz yoksa çok hızlı oluyor ve yukarı yerine aşağı komutu çıkabiliyor
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(recor_id,"down")
            time.sleep(0.1)
        elif keyboard.is_pressed("right"): # hiçbir şey yapmaması gerektiği durumda ileri tuşuna basacağım ki yanlışlıkla zıplama gibi durumlar olmasın ve normal zamanda ekranın akması gerektiğini bilsin
            record_screen(recor_id,"right")
            time.sleep(0.1)
    except RuntimeError: continue
