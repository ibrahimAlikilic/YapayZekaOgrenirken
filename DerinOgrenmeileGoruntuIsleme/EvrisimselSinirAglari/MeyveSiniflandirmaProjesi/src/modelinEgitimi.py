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
        
    # np.array olarak döndürmüyoruz çünkü ilk önce preProcess uygulamamız lazım ardından np.array haline gelmeli
    return images, labels

#########################################################

# Veri seti yolunu ayarlayalım
train_folder = "../input/archive/fruits-360_dataset_original-size/fruits-360-original-size/Training"
test_folder = "../input/archive/fruits-360_dataset_original-size/fruits-360-original-size/Test"
validation_folder = "../input/archive/fruits-360_dataset_original-size/fruits-360-original-size/Validation"

# Train, test, validation setlerini oluşturalım # boyutlar için 100,100 diye bir not okudum o yüzden 100 yolluyorum
x_train, y_train = KlasordenResimiYukle(train_folder,(100,100))
x_test, y_test = KlasordenResimiYukle(test_folder,(100,100))
x_validation, y_validation = KlasordenResimiYukle(validation_folder,(100,100))
"""
# Şimdi boyutları görelim
print("x_train boyutu:", x_train.shape)
print("y_train boyutu:", y_train.shape)
print("x_test boyutu:", x_test.shape)
print("y_test boyutu:", y_test.shape)
print("x_validation boyutu:", x_validation.shape)
print("y_validation boyutu:", y_validation.shape)
"""
#########################################################

# proprecess ( süreç ):
def preProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)# histogramı 0-255 arasına genişletme 
    img=img/255
    return img

#########################################################

# Tüm verimize preProcess uygulayalım
x_train = np.array([preProcess(img) for img in x_train])
x_test = np.array([preProcess(img) for img in x_test])
x_validation = np.array([preProcess(img) for img in x_validation])


# eğitime hazır hale getirmek için yeiden boyutlandıralım 
x_train=x_train.reshape(-1,100,100,1)
x_test=x_test.reshape(-1,100,100,1)
x_validation=x_validation.reshape(-1,100,100,1)

print("***************************************************************")
print("x_train boyutu:", x_train.shape)
print("y_train boyutu:", len(y_train))  # y_train bir liste olduğu için len() kullanıyoruz
print("x_test boyutu:", x_test.shape)
print("y_test boyutu:", len(y_test))
print("x_validation boyutu:", x_validation.shape)
print("y_validation boyutu:", len(y_validation))

#########################################################

# Data Generate (Veri oluştur)
dataGen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.1,
                           rotation_range=10)
dataGen.fit(x_train)

#########################################################

# noOfClasses lazım olacak onun için fonksiyon
def LabelSayisi(folder_path):
    # Resimlerimiz ve sınıflarımız için listeler oluşturalım
    images = []
    labels = []
    class_folders = os.listdir(folder_path)
    noOfClasses=len(class_folders)
    return noOfClasses
noOfClasses=LabelSayisi(train_folder) # dosya var olduğunu bildiğimden 1 tanesi yeterli

#########################################################

# Bazı verileri categorical hale getireceğiz, keras için bu gerekli
y_train=to_categorical(y_train,noOfClasses)
y_test=to_categorical(y_test,noOfClasses)
y_validation=to_categorical(y_validation,noOfClasses)

#########################################################

# Modelimizi inşa etme aşamasına gelmiş bulunmaktayız
model=Sequential()
model.add(Conv2D(input_shape=(100,100,1),filters=8,kernel_size=(5,5),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))# ezberlemeyi engelliyoruz
model.add(Flatten())
model.add(Dense(units=256,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=noOfClasses,activation="softmax"))

# modeli compile etmek için gerekli olan parametrelerimiz
model.compile(loss="categorical_crossentropy",optimizer=("Adam"),metrics=["accuracy"])
batch_size=250