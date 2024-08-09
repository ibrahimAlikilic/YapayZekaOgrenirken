from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss

mon = {"top":441 , "left":721 , "width":225 , "height":80 } 
sct = mss()

width=125
height=50

# model yükle
model=model_from_json(open("model_new.json","r").read())
model.load_weights("trex_weight.h5")

# etiketlediklerimiz : down=0 , right=1 , up=2
labels=["Down","Right","Up"]

framerate_time=time.time()
counter=0
i=0
delay=0
key_down_pressed=False

while True:
    img=sct.grab(mon) 
    im=Image.frombytes("RGB",img.size,img.rgb)
    im2=np.array(im.convert("L").resize((width,height)))
    im2=im2/255
    X=np.array([im2]) # bence im2 videoda im kullanmış çıktıya göre bakarsın
    X=X.reshape(X.shape[0],width,height,1)
    r=model.predict(X) # fonksiyonu, bu giriş verisine dayanarak modelin çıkış (tahmin (predict)) yapmasını sağlar. # toplamı 1 olan 2 tane sayıdan oluşuyor
    result= np.argmax(r) # r listesinin içinde bulunanların en büyük değerin indexini döndürür
    # labels lara baktığımda index=0 ise aşağı index =2 ise yukarı demek ona göre kodumuzu düzenleyeceğiz
    if result==0: # down=0
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed=True
    elif result==2:
        # öncesinde aşağı tuşuna basıldıysa serbest bırakmamız lazım
        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
        time.sleep(delay)
        keyboard.press(keyboard.KEY_UP)
        # yukarı bastıktan sonrak kısa bir süre basılı kalmasını istiyoruz
        if i <1500: # 1500. frame den sonra oyun hızlanıyormuş o yüzden 1500
            time.sleep(0.3) # sonuçta 30ms uyuyor
        elif i<5000:
            time.sleep(0.2)
        else:
            time.sleep(0.17)
        
        # yukarı bastık , bekledik , sonra tekrardan düzgün hale getirmek için aşağı tuşuna basmamız lazımmış 
        # ve tuş basılı kalıyormuş o yüzden release diyeceğiz
        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)
    counter+=1
    if (time.time()-framerate_time)>1:
        counter=0
        framerate_time=time.time()
        if i<=1500:
            delay-=0.003
        else:
            delay-=0.005
        if delay==0:
            delay=0
        print("-----------------------")
        print("Down {}\nRight {}\nUp {}\n".format(r[0][0],r[0][1],r[0][2]))
        i+=1