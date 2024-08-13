import numpy as np
import cv2
import os # veriyi içeri aktaracağız
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns # görselleştirme için
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

#########################################################

"""
Veri setini inceledim, tamamı test, train, validation olarak ayrılmış. Bu durumda fonksiyon oluşturup içlerine girip boyut eşitleyip np.array olarak geri döndürebilirim.
"""
def KlasordenResimiYukle(folder_path, img_size=(32, 32)):
    # Resimlerimiz ve sınıflarımız için listeler oluşturalım
    images = []
    labels = []
    class_folders = os.listdir(folder_path)  # Ana klasör içindeki sınıf klasörlerinin isimlerini listeleyelim
    
    # Her bir klasörü gezelim
    for class_folder in class_folders:
        # Klasörün tam yolunu oluşturmak için os.path.join kullanmalıyız
        class_path = os.path.join(folder_path, class_folder)
        
        # Klasör adı, sınıfın etiketi olarak kullanılır (örneğin, 'apple_red_2', 'banana_yellow' gibi)
        class_label = class_folder  # Klasör adını direkt etiket olarak kullanıyoruz
        
        # Şimdi bu sınıf klasörünün içindeki tüm görüntü dosyalarını listeleyelim
        for image_name in os.listdir(class_path):
            # Her bir görüntü dosyasının tam yolunu oluşturuyoruz
            image_path = os.path.join(class_path, image_name)
            
            image = cv2.imread(image_path)
            
            if image is not None:
                image = cv2.resize(image, img_size)
                
                # Şimdi görüntümüzü labels ve images listelerimize ekleyelim
                images.append(image)
                labels.append(class_label)
        
    # Biliyoruz ki biz bu verileri np.array olarak kullanıyoruz, o yüzden dönüşümü yaparak geri döndürelim
    return np.array(images), np.array(labels)

#########################################################

# Veri seti yolunu ayarlayalım
train_folder = "../input/archive/fruits-360_dataset_original-size/fruits-360-original-size/Training"
test_folder = "../input/archive/fruits-360_dataset_original-size/fruits-360-original-size/Test"
validation_folder = "../input/archive/fruits-360_dataset_original-size/fruits-360-original-size/Validation"

# Train, test, validation setlerini oluşturalım # boyutlar için 100,100 diye bir not okudum o yüzden 100 yolluyorum
x_train, y_train = KlasordenResimiYukle(train_folder,(100,100))
x_test, y_test = KlasordenResimiYukle(test_folder,(100,100))
x_validation, y_validation = KlasordenResimiYukle(validation_folder,(100,100))

# Şimdi boyutları görelim
print("x_train boyutu:", x_train.shape)
print("y_train boyutu:", y_train.shape)
print("x_test boyutu:", x_test.shape)
print("y_test boyutu:", y_test.shape)
print("x_validation boyutu:", x_validation.shape)
print("y_validation boyutu:", y_validation.shape)
