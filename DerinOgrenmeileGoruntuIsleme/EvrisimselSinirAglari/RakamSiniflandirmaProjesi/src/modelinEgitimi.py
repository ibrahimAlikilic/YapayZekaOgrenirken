import numpy as np
import cv2
import os # veriyi içeri aktaracağız
from sklearn.model_selection import train_test_split
import seaborn as sns # göreslleştirme için
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras import layers
from keras.layers import Dense,Conv1D, MaxPooling2D, Flatten , Dropout , BatchNormalization
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
    img=igm/255
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

#########################################################
