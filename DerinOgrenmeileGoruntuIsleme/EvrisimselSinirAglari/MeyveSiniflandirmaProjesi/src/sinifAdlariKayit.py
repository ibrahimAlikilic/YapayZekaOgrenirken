# tespit edilen nesneyi index değil de isim olarak görüntülemek için böyle bir dosya oluşturmaya karar verdim
import os
import pickle
from sklearn.preprocessing import LabelEncoder

def save_class_names(train_folder, file_path):
    # Sınıf adlarını saklayacak bir dictionary oluşturalım
    class_folders = os.listdir(train_folder)
    class_names = sorted(class_folders)  # Sınıf isimlerini sıralı hale getirebiliriz
    
    # class_names listesini kaydedelim
    with open(file_path + "/class_names.pkl", "wb") as pickle_out:
        pickle.dump(class_names, pickle_out)

    print("Sınıf adları başarıyla kaydedildi.")

# Veri seti yolunu ve çıkış dosyasının yolunu ayarlayalım
train_folder = "../input/archive/Training"
file_path = "../output"

# Klasörün var olup olmadığını kontrol edin, yoksa oluşturun
if not os.path.exists(file_path):
    os.makedirs(file_path)

# Fonksiyonu çağırarak sınıf adlarını kaydedelim
save_class_names(train_folder, file_path)