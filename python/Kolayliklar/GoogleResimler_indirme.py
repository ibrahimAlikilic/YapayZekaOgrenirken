'''
window da chrome kullanman lazım diğer türlü olmaz.
'''

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
import requests

# Google Görseller URL'sini oluştur
query = ""  # Buraya arama yapmak istediğiniz kelimeyi girin
url = f"https://www.google.com/search?q={query}&source=lnms&tbm=isch"

# Webdriver'ı başlat
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(url)

# Sayfanın yüklenmesini bekleyin
time.sleep(2)

input("Sayfa hazır mı ? : ") # sayfayı en sona getirene kadar zaman kazanmak için bu soruyu sordum.

# Tüm resim öğelerini bulun
images = driver.find_elements(By.CSS_SELECTOR, "img")

# Resimleri kaydedileceği klasörü oluşturun
save_path = "images"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Resimleri indir
for idx, img in enumerate(images):
    try:
        img_url = img.get_attribute('src')
        
        # URL'in geçerli olup olmadığını kontrol edin
        if img_url and img_url.startswith('http'):
            img_data = requests.get(img_url).content
            with open(f"{save_path}/{query}_{idx}Bir.jpg", 'wb') as handler:
                handler.write(img_data)
            print(f"{query}_{idx}.jpg indirildi.")
        else:
            print(f"Geçersiz URL tespit edildi: {img_url}")
    
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

# Webdriver'ı kapat
driver.quit()
