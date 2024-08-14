import numpy as np
import cv2
import os # veriyi içeri aktaracağız
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns # görselleştirme için
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input
from tensorflow.keras import layers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

#########################################################

"""
Veri setini inceledim, tamamı test, train olarak ayrılmış. Bu durumda fonksiyon oluşturup içlerine girip boyut eşitleyip images, labels olarak döndürebilirim
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
train_folder = "../input/archive/Training"
test_folder = "../input/archive/Test"

# Train, test, validation setlerini oluşturalım # boyutlar için 32,32 diye bir not okudum o yüzden 100 yolluyorum
# 100 sistem kaldırmadı o yüzden 32,32 gönderdim
x_train, y_train = KlasordenResimiYukle(train_folder,(32,32))
x_test, y_test = KlasordenResimiYukle(test_folder,(32,32))
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=0.2,random_state=42)
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
x_train=x_train.reshape(-1,32,32,1)
x_test=x_test.reshape(-1,32,32,1)
x_validation=x_validation.reshape(-1,32,32,1)

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
# Etiketleri sayısal değerlere dönüştürmek için LabelEncoder kullanıyoruz
label_encoder = LabelEncoder()

# String etiketleri sayısal değerlere çeviriyoruz
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_validation = label_encoder.transform(y_validation)

# Categorical hale getirme
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

#########################################################

# Modelimizi inşa etme aşamasına gelmiş bulunmaktayız
model = Sequential()
model.add(Input(shape=(32,32,1)))
model.add(Conv2D(filters=8, kernel_size=(5,5), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=noOfClasses, activation="softmax"))

# modeli compile etmek için gerekli olan parametrelerimiz
model.compile(loss="categorical_crossentropy",optimizer=("Adam"),metrics=["accuracy"])
batch_size=250

#########################################################

# şimdi eğitim aşamasına geçiyoruz
hist = model.fit(dataGen.flow(x_train, y_train, batch_size=batch_size),
                 validation_data=(x_validation, y_validation),
                 epochs=76, steps_per_epoch=x_train.shape[0] // batch_size,
                 shuffle=True)

# epochs=81 denedim ve oluşan ilk önce oluşan grafiklere baktım ardından uçbirim ekranındaki sonuçlarla teğit ettim ve epochs=76 en uygun olduğuna karar verdim.
#########################################################

# Kayıt

# Klasörün var olup olmadığını kontrol edin, yoksa oluşturun
file_path = "../output"
if not os.path.exists(file_path):
    os.makedirs(file_path)

# Modeli kaydedin
with open(file_path + "/model_trained_new.p", "wb") as pickle_out:
    pickle.dump(model, pickle_out)
print("Model kaydedildi.")

# Çıkan sonucumuzu görselleştirerek değerlendirelim

# Eğitim süreci sırasında elde edilen metriklerin anahtarlarını görmek için
print(hist.history.keys())

# Eğitim ve doğrulama kayıp değerlerini çizmek için
plt.figure()
plt.plot(hist.history["loss"], label="Eğitim Loss")
plt.plot(hist.history["val_loss"], label="Val Loss")
plt.legend()
plt.savefig(os.path.join(file_path, "loss_plot.png"))
print("Loss Kaydedildi")
plt.show()

# Eğitim ve doğrulama doğruluk değerlerini çizmek için
plt.figure()
plt.plot(hist.history["accuracy"], label="Eğitim Accuracy")
plt.plot(hist.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.savefig(os.path.join(file_path, "accuracy_plot.png"))
print("Accuracy Kaydedildi")
plt.show()

#########################################################

# Sonuçlara bakma

score=model.evaluate(x_test,y_test,verbose=1) # Bu kod satırı, modelinizin test veri kümesi üzerinde performansını değerlendirir. 
#verbose=1: Bu, değerlendirme sırasında ilerlemeyi gösterecek ayrıntılı çıktıyı sağlar. verbose=0 sessiz bir çıktı verirken, verbose=2 ise daha fazla ayrıntı gösterir.
print("Test loss : ",score[0])
print("Test accuracy : ",score[1])

y_pred=model.predict(x_validation)
y_pred_class=np.argmax(y_pred,axis=1)
Y_true=np.argmax(y_validation,axis=1)
cm=confusion_matrix(Y_true,y_pred_class)
# aşağıda oluşturduğumuz görsel sayesinde doğru tahminimizi görmüş oluyoruz
f,ax=plt.subplots(figsize=(35,35))
sns.heatmap(cm,annot=True,linewidths=0.1,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)
plt.xlabel("Predicted")
plt.ylabel("true")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(file_path, "confusion_matrix.png"))
print("Confusion Matrix kaydedildi")
plt.show()