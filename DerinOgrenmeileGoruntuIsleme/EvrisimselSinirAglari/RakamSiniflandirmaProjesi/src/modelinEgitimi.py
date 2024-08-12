import numpy as np
import cv2
import os # veriyi içeri aktaracağız
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns # göreslleştirme için
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense,Conv2D, MaxPooling2D, Flatten , Dropout , BatchNormalization
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle # modeli yüklemek için kullanacağız

#########################################################

# verimizi inceledik baktık hepsi hemen hemen aynı boyutta e bu tanımayı azaltır bu yüzden zoom in-out yapacağız
path="../input/myData"
myList=os.listdir(path)
noOfClasses=len(myList)
print("Label sayısı : ",noOfClasses)


images=[]
classNo=[] # etiketimiz

#########################################################

for i in range(noOfClasses): # noOfClasses uzunluğu kadar dön
    myImageList=os.listdir(path+"//"+str(i)) # i dedik çünkü dosya adlarım zaten 0-10 ( i=label larım oldu +=classNO aşağıda yaptık. )
    #şimdi klasörlerin içerisine girelim
    for j in myImageList:
        img=cv2.imread(path+"//"+str(i)+"//"+j)
        # eğiteceğimiz nörel şeyin girdi boyutu 32X32 imiş o yüzden resize yapalım
        img=cv2.resize(img,(32,32))
        images.append(img)
        classNo.append(i)
print("images adet : ",len(images))
print("classNo adet : ",len(classNo))

# bundan sonra veriler numpy dizisi olarak lazım o yüzden dönüşüm yapalım
images=np.array(images)
classNo=np.array(classNo)
print("images boyutu : ",images.shape)
print("classNo boyutu : ",classNo.shape) # classNO vektör olduğunan 10160, şeklinde çıktı alıyoruz

#########################################################

# veri ayırma
x_train,x_test,y_train,y_test=train_test_split(images,classNo,test_size=0.5,random_state=42) # 1.
x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=0.2,random_state=42) # 2.
"""
1. kod açıklama :
Amaç: images ve classNo olarak adlandırılan veri setlerini, %50'si eğitim için (x_train, y_train), %50'si test için (x_test, y_test) olacak şekilde 
ikiye böler.

Parametreler:

    images: Girdi görüntüleri. Bu, modelin öğrenmesini sağlayacak verilerin olduğu yer.
    classNo: Hedef etiketleri. Bu, her görüntünün sınıfını (örneğin, bir kedi mi yoksa köpek mi olduğunu) belirten etiketlerdir.
    test_size=0.5: Veri setinin %50'sinin test veri seti olarak ayrılacağını belirtir.
    random_state=42: Kodun tekrarlanabilirliğini sağlamak için kullanılır. Aynı random_state değeri verildiğinde, her seferinde aynı şekilde veri seti bölünür.

Sonuç:

    x_train: Eğitim veri setine ait görüntüler.
    x_test: Test veri setine ait görüntüler.
    y_train: Eğitim veri setine ait sınıf etiketleri.
    y_test: Test veri setine ait sınıf etiketleri.
"""
"""
2. kod açıklama :
Amaç: İlk satırda elde edilen x_train ve y_train veri setini, %20'si doğrulama için (x_validation, y_validation), 
%80'i eğitim için (x_train, y_train) olacak şekilde tekrar böler.

Parametreler:

    x_train: Eğitim veri setine ait görüntüler (ilk satırda oluşturulan).
    y_train: Eğitim veri setine ait sınıf etiketleri (ilk satırda oluşturulan).
    test_size=0.2: Eğitim veri setinin %20'sinin doğrulama veri seti olarak ayrılacağını belirtir.
    random_state=42: Yine aynı şekilde veri setinin aynı şekilde bölünmesi için kullanılır.

Sonuç:

    x_train: Bu satırdan sonra, orijinal veri setinin %40'ını içerir (ilk %50'lik eğitim setinin %80'i).
    x_validation: Bu satırdan sonra, orijinal veri setinin %10'unu içerir (ilk %50'lik eğitim setinin %20'si).
    y_train: x_train için karşılık gelen sınıf etiketleri.
    y_validation: x_validation için karşılık gelen sınıf etiketleri.
"""
"""
Özet:

    İlk Bölme: Veri setinin %50'si eğitim, %50'si test için ayrılır.
    İkinci Bölme: Eğitim için ayrılan veri setinin %80'i asıl eğitim için, %20'si doğrulama için kullanılır.

Sonuç olarak:

    Eğitim seti (x_train, y_train) veri setinin %40'ını,
    Doğrulama seti (x_validation, y_validation) %10'unu,
    Test seti (x_test, y_test) ise %50'sini içerir.
"""

# train ve validation ları kullanarak eğitime yapacağız ardından test ler ile test edeceğiz
print("x_train boyutu : ",x_train.shape)
print("x_test boyutu : ",x_test.shape)
print("x_validation boyutu : ",x_validation.shape)
# yukarıda aldığım çıktı sayesinde verimin kaç tanesinin ne için ayrılmış olduğunu görüm .

#########################################################
""" Çok vakit alıyorr yorum satırı kalsın
# Görselleştirme yapacağız 

fig,axes=plt.subplots(3,1,figsize=(7,7))
fig.subplots_adjust(hspace=0.5)
sns.countplot(y_train,ax=axes[0])
axes[0].set_title("y_train")

sns.countplot(y_test,ax=axes[1])
axes[1].set_title("y_test")

sns.countplot(y_validation,ax=axes[2])
axes[2].set_title("y_validation")
plt.show()
"""

"""
Yukarıdaki eğitimdeki kod aşağıdaki ise benim gpt ile yazdığım kod :

# Sınıf dağılımlarını hesapla
train_classes, train_counts = np.unique(y_train, return_counts=True)
test_classes, test_counts = np.unique(y_test, return_counts=True)
validation_classes, validation_counts = np.unique(y_validation, return_counts=True)

# 1. Büyük pencere, 3 ayrı tablo
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# y_train grafiği
axs[0].bar(train_classes, train_counts, color='#FF6347')
axs[0].set_title('y_train Distribution')
axs[0].set_xlabel('Classes')
axs[0].set_ylabel('Count')

# y_test grafiği
axs[1].bar(test_classes, test_counts, color='#4682B4')
axs[1].set_title('y_test Distribution')
axs[1].set_xlabel('Classes')
axs[1].set_ylabel('Count')

# y_validation grafiği
axs[2].bar(validation_classes, validation_counts, color='#8FBC8F')
axs[2].set_title('y_validation Distribution')
axs[2].set_xlabel('Classes')
axs[2].set_ylabel('Count')

# Aralarındaki boşlukları ayarla
plt.tight_layout()

# 3 ayrı tabloyu göster ve kaydet
plt.savefig('../output/class_distributions_separate.png')
plt.show()

# 2. Tek pencerede birleştirilmiş tablo
plt.figure(figsize=(10, 6))

width = 0.25  # Barların genişliği
x = np.arange(len(train_classes))  # X ekseni için sınıf sayısı kadar aralık

# Bar grafikleri oluştur
plt.bar(x - width, train_counts, width, color='#FF6347', label='y_train')
plt.bar(x, test_counts, width, color='#4682B4', label='y_test')
plt.bar(x + width, validation_counts, width, color='#8FBC8F', label='y_validation')

# Etiketler ve başlıklar ekle
plt.xlabel('Classes')
plt.ylabel('Count')
plt.title('Combined Distribution of Classes in y_train, y_test, y_validation')
plt.xticks(x, train_classes)  # X eksenine sınıfları ekle
plt.legend(title='Datasets')

# Tek tabloyu göster ve kaydet
plt.savefig('../output/class_distributions_combined.png')
plt.show()
"""

#########################################################

# proprecess ( süreç ):
def preProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)# histogramı 0-255 arasına genişletme 
    img=img/255
    return img
"""
Fonksiyonun anlatımı :

Bu preProcess fonksiyonu, bir görüntüyü ön işlemden geçirmek için kullanılıyor. Aşağıda adım adım açıklamalarını bulabilirsin:

    Gri Tonlamaya Çevirme (cv2.cvtColor):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) satırı, verilen renkli görüntüyü gri tonlamalı bir görüntüye çevirir.
        Bu işlem, üç renk kanalını (B, G, R) tek bir kanal (gri ton) haline getirir. Böylece, görüntünün parlaklık bilgisi korunur.

    Histogram Eşitleme (cv2.equalizeHist):
        img=cv2.equalizeHist(img) satırı, görüntünün histogramını eşitler. Histogram eşitleme, görüntüdeki kontrastı artırmak için kullanılır.
        Bu işlem, piksel değerlerini 0-255 aralığında genişleterek, görüntünün daha belirgin hale gelmesini sağlar. Özellikle düşük kontrastlı görüntülerde bu işlem çok faydalı olabilir.

    Normalizasyon:
        img=igm/255 satırı, piksel değerlerini 0-1 aralığına ölçekler.
        Bu, genellikle derin öğrenme modellerinde giriş verilerini normalize etmek için yapılan bir işlemdir.
        Bu şekilde, veriler modelin daha verimli çalışmasını sağlar.

    Geri Döndürme:
        return img satırı, işlenmiş görüntüyü geri döndürür.

Bu fonksiyon, görüntüyü gri tonlamaya çevirerek, kontrastını artırarak ve normalleştirerek model için daha uygun hale getirir.
Bu tür ön işlemler, görüntü tanıma ve sınıflandırma gibi görevlerde modelin daha iyi performans göstermesine yardımcı olabilir.
"""

# preProcess işlemini tüm verimize yapacağız bunun için de map yüntemini kullanacağız.
x_train=np.array(list(map(preProcess,x_train)))
x_test=np.array(list(map(preProcess,x_test)))
x_validation=np.array(list(map(preProcess,x_validation)))
# map fonksiyonu, x_train dizisindeki her bir elemanı preProcess işlevine uygulayarak bir işlem yapar. 
# Yani, preProcess fonksiyonu, x_train içindeki her bir öğe için çağrılır.

# eğitime hazır hale getirmek için yeiden boyutlandıralım
x_train=x_train.reshape(-1,32,32,1)
x_test=x_test.reshape(-1,32,32,1)
x_validation=x_validation.reshape(-1,32,32,1)

# -1: Bu, NumPy'ye mevcut öğe sayısına göre bu boyutun otomatik olarak hesaplanmasını söyler. 
# Yani, toplam öğe sayısı, diğer boyutlar (32, 32, ve 1) ile uyumlu olacak şekilde hesaplanır.

#########################################################

# Data Generate (Veri oluştur)
dataGen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.1,
                           rotation_range=10)
"""
Bu kod, ImageDataGenerator sınıfını kullanarak veri artırma (data augmentation) işlemleri tanımlar. İşte her bir parametrenin anlamı:

width_shift_range=0.1:
Bu, görüntülerin yatay olarak %10 kadar kaydırılmasına izin verir. Yani, görüntüler yatay eksende %10 kadar sola veya sağa kaydırılabilir.

height_shift_range=0.1:
Bu, görüntülerin dikey olarak %10 kadar kaydırılmasına izin verir. Yani, görüntüler dikey eksende %10 kadar yukarı veya aşağı kaydırılabilir.

zoom_range=0.1:
Bu, görüntülerde %10 kadar yakınlaştırma veya uzaklaştırma yapılmasına izin verir. Yani, görüntüler biraz daha yakınlaştırılabilir veya biraz uzaklaştırılabilir.

rotation_range=10:
Bu, görüntülerin 10 dereceye kadar döndürülebilmesine izin verir. Yani, görüntüler rastgele 10 dereceye kadar döndürülür.

Bu ayarlarla, modelinizin daha çeşitli veri ile eğitilmesini sağlayarak genelleme yeteneğini artırabilirsiniz. 
Bu teknikler, modelinizin overfitting (aşırı uyum) yapmasını önlemeye yardımcı olabilir.
"""
dataGen.fit(x_train)
# Bu satır, ImageDataGenerator nesnesinin fit yöntemini kullanarak x_train veri kümesini fit eder. 
# Bu işlem, veri artırma işlemleri sırasında kullanılması gereken istatistikleri hesaplar. 

#########################################################

# Bazı verileri categorical hale getireceğiz, keras için bu gerekli

y_train=to_categorical(y_train,noOfClasses)
y_test=to_categorical(y_test,noOfClasses)
y_validation=to_categorical(y_validation,noOfClasses)
"""
to_categorical(y_train, noOfClasses): Bu fonksiyon, y_train'deki her tam sayıyı, noOfClasses kadar olan one-hot encoded bir vektöre dönüştürür. 
noOfClasses, sınıfların toplam sayısını belirtir.

Örneğin, eğer noOfClasses 3 ise ve y_train'deki bir etiket 1 ise, bu etiket [0, 1, 0] şeklinde one-hot encoded bir vektöre dönüşür.
"""

#########################################################

# Modelimizi inşa etme aşamasına gelmiş bulunmaktayız
model=Sequential()
model.add(Conv2D(input_shape=(32,32,1),filters=8,kernel_size=(5,5),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16,kernel_size=(3,3),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))# ezberlemeyi engelliyoruz
model.add(Flatten())
model.add(Dense(units=256,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=noOfClasses,activation="softmax"))

"""
Bu kod, bir sinir ağı modelini tanımlar ve yapılandırır. Model, Keras'ın Sequential API'si kullanılarak oluşturulmuştur ve temel bir konvolüsyonel sinir ağı (CNN) yapısını içerir. İşte her bir adımın açıklaması:

    Model Tanımı:

model = Sequential()

Sequential API, katmanları sırayla eklemeye olanak tanır.

İlk Konvolüsyonel Katman:

model.add(Conv2D(input_shape=(32, 32, 1), filters=8, kernel_size=(5, 5), activation="relu", padding="same"))

    input_shape=(32, 32, 1): Giriş verisinin boyutları; 32x32 piksel boyutunda ve 1 renk kanalı (gri tonlamalı).
    filters=8: 8 adet filtre (veya kernel) kullanılır.
    kernel_size=(5, 5): Filtre boyutu 5x5 piksel.
    activation="relu": ReLU aktivasyon fonksiyonu kullanılır.
    padding="same": Giriş ve çıkış boyutlarının aynı kalmasını sağlar.

İlk Maksimum Havuzlama Katmanı:

model.add(MaxPooling2D(pool_size=(2, 2)))

    pool_size=(2, 2): 2x2 piksel havuzlama yapılır. Bu, özellik haritalarının boyutunu yarıya indirir.

İkinci Konvolüsyonel Katman:

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same"))

    filters=16: 16 adet filtre kullanılır.
    kernel_size=(3, 3): Filtre boyutu 3x3 piksel.

İkinci Maksimum Havuzlama Katmanı:

model.add(MaxPooling2D(pool_size=(2, 2)))

    Aynı havuzlama boyutu kullanılır ve özellik haritalarının boyutunu tekrar yarıya indirir.

Dropout Katmanı:

model.add(Dropout(0.2))

    Dropout(0.2): Eğitim sırasında nöronların %20'sini rastgele kapatarak overfitting'i (aşırı uyum) azaltır.

Flatten Katmanı:

model.Flatten()

    Bu, 2D özellik haritalarını 1D vektöre dönüştürür, böylece yoğun bağlantılı (dense) katmanlara beslenebilir hale getirir.

İlk Yoğun Katman:

model.add(Dense(units=256, activation="relu"))

    units=256: 256 nöron içerir.
    activation="relu": ReLU aktivasyon fonksiyonu kullanılır.

İkinci Dropout Katmanı:

model.add(Dropout(0.2))

    Yine %20 dropout uygulanır.

Çıkış Katmanı:

    model.add(Dense(units=noOfClasses, activation="softmax"))

        units=noOfClasses: Toplam sınıf sayısına eşit sayıda nöron içerir.
        activation="softmax": Sınıflandırma problemlerinde olasılıkları hesaplamak için kullanılır.

Bu model, genellikle görüntü sınıflandırma görevlerinde kullanılır ve temel bir konvolüsyonel sinir ağı (CNN) yapısına sahiptir.
"""

#########################################################

# modeli compile etmek için gerekli olan parametrelerimiz
model.compile(loss="categorical_crossentropy",optimizer=("Adam"),metrics=["accuracy"])
batch_size=250

#########################################################

# şimdi eğitim aşamasına geçiyoruz
hist = model.fit(dataGen.flow(x_train, y_train, batch_size=batch_size),
                 validation_data=(x_validation, y_validation),
                 epochs=15, steps_per_epoch=x_train.shape[0] // batch_size,
                 shuffle=True)
"""
dataGen.flow(x_train, y_train, batch_size=batch_size): 
Bu, ImageDataGenerator'ın flow yöntemiyle veri artırma işlemleri uygulayarak eğitim verilerini döndürür.
batch_size ile her seferinde kaç örneğin işleneceğini belirler.

validation_data=(x_validation, y_validation):
Modelin doğrulama verileri ile performansını değerlendirmek için kullanılır.
x_validation ve y_validation doğrulama veri kümesinin özellikleri ve etiketleridir.

epochs=15: Modelin toplamda 15 dönem (epoch) boyunca eğitileceğini belirtir.

steps_per_epoch=x_train.shape[0] // batch_size:
Her dönemde kaç adım (batch) yapılacağını belirtir. Eğitim veri kümesindeki örnek sayısının batch_size'a bölünmesiyle elde edilir.

shuffle=True: Eğitim sırasında verilerin karıştırılmasını sağlar.
Bu, modelin genel performansını artırabilir çünkü veriler her seferinde farklı bir sırada sunulur.
"""

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
"""
Bu kod parçacığı, modelinizin doğrulama veri kümesi üzerindeki tahminlerini değerlendirir ve bir karışıklık matrisi (confusion matrix) oluşturur.
İşte adım adım açıklamalar:

    Tahminlerin Yapılması:

y_pred = model.predict(x_validation)

    model.predict(x_validation): Modeliniz, doğrulama veri kümesindeki x_validation örnekleri için tahminlerde bulunur.
    y_pred, tahmin edilen olasılıkları içeren bir dizidir.

Tahminlerin Sınıflara Dönüştürülmesi:

y_pred_class = np.argmax(y_pred, axis=1)

    np.argmax(y_pred, axis=1): 
    y_pred dizisindeki her tahmin için en yüksek olasılığa sahip sınıfı seçer. Bu, her örnek için tahmin edilen sınıf etiketlerini verir.

Gerçek Etiketlerin Sınıflara Dönüştürülmesi:

Y_true = np.argmax(y_validation, axis=1)

    np.argmax(y_validation, axis=1): 
    Gerçek etiketlerin one-hot encoded formatından sınıf etiketlerine dönüştürülmesini sağlar.
    Y_true, her örnek için gerçek sınıf etiketlerini verir.

Karışıklık Matrisinin Hesaplanması:

cm = confusion_matrix(Y_true, y_pred_class)

    confusion_matrix(Y_true, y_pred_class):
    Gerçek etiketler (Y_true) ve tahmin edilen etiketler (y_pred_class) arasındaki ilişkiyi gösteren bir karışıklık matrisini hesaplar.
    Karışıklık matrisi, modelinizin her sınıf için doğru ve yanlış sınıflandırmaları görmenizi sağlar.
"""

# aşağıda oluşturduğumuz görsel sayesinde doğru tahminimizi görmüş oluyoruz
f,ax=plt.subplots(figsize=(8,8))
sns.heatmap(cm,annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt=".1f",ax=ax)
plt.xlabel("Predicted")
plt.ylabel("true")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(file_path, "confusion_matrix.png"))
print("Confusion Matrix kaydedildi")
plt.show()