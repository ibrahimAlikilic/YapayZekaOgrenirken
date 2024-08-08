import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") # uyarıları kale almadık dedik.

imgs=glob.glob("./output/*.png") # output klasörümdeki tüm png leri al dedik
# resimlerim için boyut belirleyeceğim
width=125
height=50
# Resimleri ve label leri eklemek için 2 tane listeye ihtiyacım olacak
X=[] # resimler
Y=[] # label lar
# döngü yapacağız ve döngümüzde resimlerimize eğitim öncesi gerekli olan process (işlem) leri gerçekleştireceğiz
# label ( etiketimizi ) down , up , right olarak ayırıp class larımızı bileceğiz
for img in imgs:
    filename=os.path.basename(img) # resmin adını aldı
    label=filename.split("_")[0] # altçizgiye göre bölüp 0. index tekini aldık.
    # şimdi resmimizi convert ( dönüştürmek ) edeceğiz yani size (boyut) sını değiştireceğiz
    im=np.array(Image.open(img).convert("L").resize((width,height)))
    """
    Image.open(img):

        Bu adımda, img ile belirtilen görüntü dosyası açılır. img değişkeni, bir dosya yolunu veya bir görüntü dosyasını temsil eder.

    .convert("L"):

        Bu kısım, görüntüyü gri tonlamalı bir görüntüye dönüştürür. "L" modu, 8 bitlik gri tonlamalı bir görüntüyü temsil eder, 
        yani her piksel 0 ile 255 arasında bir değere sahip olur.

    .resize((width, height)):

        Bu adım, görüntüyü belirtilen width (genişlik) ve height (yükseklik) boyutlarına göre yeniden boyutlandırır. 
        Bu, görüntünün boyutlarını istenilen değerlere göre ayarlar.

    np.array(...):

        Son olarak, elde edilen görüntü bir NumPy dizisine dönüştürülür. 
        Bu, görüntünün piksel verilerini içeren bir dizi oluşturur ve bu sayede bu veri üzerinde sayısal işlemler yapmak mümkün hale gelir.
    """
    # Normalize ( 0-1 arasında değerler haline ) etmem lazım
    im=im/255
    X.append(im)
    Y.append(label)
# train_test_split array kabul eder bu yüzden listemi np.array haline getireceğim
X=np.array(X)
X=X.reshape(X.shape[0],width,height,1) # kaç resim olduğu , yeni genişlik değeri , yeni yükseklik değeri , 1 = channel değeri (siyah beyaz kullanacağım demek).

# şimdi label lerim string veri tipindae ama biz biliyoruz ki int kullanıyoruz bu yüzden 0,1,2 haline getireceğim ve bunu binary olarak ifade edeceğim çünkü keras bu şekilde kabul ediyor.
def onehot_labels(values):
    label_encoder=LabelEncoder()
    integer_encoded=label_encoder.fit_transform(values) # fit_transform demek önce çğren sonra dönüştür demek. # artık sayıya dönüştüler (0,1,2).
    onehot_encoder=OneHotEncoder(sparse=False)
    # şu anda integer_encoded.shape = (369,) burada , den sonrası yok gözüküyor bunu 1 yapacağız
    integer_encoded=integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoder=onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoder # binary şeklinde 0,1,2 döndürür
Y=onehot_labels(Y)
"""fonksiyonun ayrıntılı açılaması
Bu fonksiyon, kategorik (sınıf) etiketlerini one-hot encoding formatına dönüştürür. One-hot encoding, her bir kategoriyi bir dizi içinde yalnızca bir elemanın "1" olduğu, 
diğer elemanların "0" olduğu bir formatta temsil eder. Bu format, genellikle makine öğrenmesi modelleri için kategorik verilerin daha iyi anlaşılabilir ve işlenebilir hale getirilmesi için kullanılır. Fonksiyonun işleyişini adım adım inceleyelim:
Fonksiyon Açıklaması: onehot_labels(values)

    label_encoder = LabelEncoder():
        LabelEncoder, kategorik etiketleri tamsayı (integer) değerlerine dönüştüren bir sınıf kodlayıcıdır.
        Örneğin, ["kırmızı", "mavi", "yeşil"] gibi kategorik etiketler sırasıyla [0, 1, 2] şeklinde kodlanır.

    integer_encoded = label_encoder.fit_transform(values):
        fit_transform metodu, önce values verisi üzerinden öğrenme işlemi yapar (fit), sonra bu veriyi tamsayı değerlerine dönüştürür (transform).
        Örneğin, ["kırmızı", "mavi", "yeşil"] verisi [0, 1, 2] olarak dönüştürülür.
        integer_encoded, bu işlem sonucunda elde edilen tamsayı dizisidir.

    onehot_encoder = OneHotEncoder(sparse=False):
        OneHotEncoder, tamsayı olarak kodlanmış değerleri one-hot encoding formatına dönüştürür.
        sparse=False, dönüşüm sonucunun yoğun (dense) bir matris olarak döndürülmesini sağlar, yani bir NumPy dizisi olarak döner.

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1):
        Bu satır, integer_encoded dizisini yeniden şekillendirir. Orijinal dizinin şekli (n,) (tek boyutlu bir dizi) olabilir.
        Bu satır ile dizi (n, 1) şekline getirilir. Bu, one-hot encoding işlemi için gereklidir çünkü OneHotEncoder girişi iki boyutlu bir dizi olarak bekler.

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded):
        fit_transform metodu, tamsayı kodlanmış etiketleri one-hot encoding formatına dönüştürür. Örneğin, [0, 1, 2] dizisi aşağıdaki gibi bir matrise dönüştürülür:
        [[1, 0, 0], 
        [0, 1, 0], 
        [0, 0, 1]]
        onehot_encoded, bu dönüşümden elde edilen one-hot encoded dizidir.

    return onehot_encoded:
        Fonksiyon, one-hot encoded matrisi döndürür. Bu matris, her bir kategori için bir satır ve her bir kategori için bir sütun içerir,
        ve sadece o satırın temsil ettiği kategori sütununda "1" değerine sahiptir.

Özet

Bu fonksiyon, kategorik verileri (örneğin sınıf etiketleri) one-hot encoding formatına dönüştürmek için kullanılır. 
Bu format, özellikle sınıflandırma problemlerinde makine öğrenmesi modellerine giriş olarak verilirken tercih edilir.
"""

# şimdi train_test_split aşamasına geldik.
train_X,test_X,train_y,test_y= train_test_split(X,Y,test_size=0.25,random_state=2)
"""
Bu satır, veri setinizi eğitim (train) ve test veri setlerine böler.
Bu işlem, makine öğrenmesi modelinizi eğitmek ve daha sonra modelin performansını test etmek için kullanılır. Satırı ve parametrelerini adım adım açıklayalım:
Kodun Açıklaması :

    train_test_split(X, Y, test_size=0.25, random_state=2):
        train_test_split: Bu fonksiyon, veri setini rastgele olarak eğitim ve test setlerine böler.
        X özellik (girdi) veri seti, Y ise hedef (çıktı) etiketlerini temsil eder.

        X: Özellik veri seti. Modelin eğitileceği girdileri içerir (örneğin, görüntü verileri, özellik vektörleri).
        Y: Hedef etiketleri. Modelin tahmin etmeyi öğreneceği çıktı değerlerini içerir (örneğin, sınıf etiketleri).
        test_size=0.25: Veri setinin %25'i test veri seti olarak ayrılır, geri kalan %75'i ise eğitim veri seti olarak kullanılır.
        random_state=2: Bu parametre, bölme işlemini rastgele fakat tekrarlanabilir hale getirir.
        Aynı random_state değeri kullanıldığında, her seferinde aynı şekilde bölme işlemi yapılır. Bu, deneylerinizi tekrarlarken aynı sonuçları almanızı sağlar.

    Çıktılar:
        train_X: Eğitim veri seti için özellikler (girdiler).
        test_X: Test veri seti için özellikler (girdiler).
        train_y: Eğitim veri seti için hedef etiketler (çıktılar).
        test_y: Test veri seti için hedef etiketler (çıktılar).

Özet :
Bu kod, veri setinizi eğitim ve test setlerine böler.
Eğitim seti (train_X, train_y) modelinizi eğitmek için kullanılırken, test seti (test_X, test_y) modelinizin performansını değerlendirmek için kullanılır.
Bu yaklaşım, modelin yeni, görülmemiş verilere karşı nasıl performans göstereceğini tahmin etmeye yardımcı olur.
random_state parametresi, veri setinin bölünmesini tekrarlanabilir kılmak için kullanılır.
"""

# şimdi de cnn modelimizi inşa edeceğiz
model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(width,height,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
# piksel ekleme
model.add(MaxPooling2D(pool_size=(2,2)))
# seyreltme ekleme
model.add(Dropout(0.25))
# Düzleştirelim
model.add(Flatten())
# şimdi sınıflandırma
model.add(Dense(128,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(3,activation="softmax"))

""" CNN model inşaatı özeti
Bu kod bloğu, bir Convolutional Neural Network (CNN) modelini TensorFlow/Keras kullanarak tanımlar. 
NN'ler, özellikle görüntü işleme ve bilgisayarla görme (computer vision) alanlarında etkili olan derin öğrenme mimarileridir.
Modelin yapısını ve her bir katmanını adım adım inceleyelim:
CNN Modeli İnşası

    model=Sequential():
        Sequential sınıfı, katmanların sıralı olarak eklenebileceği bir model oluşturur.
        Bu, genellikle en basit model türüdür, çünkü katmanlar ardışık olarak birbirine eklenir.

    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(width, height, 1))):
        Conv2D: İki boyutlu konvolüsyonel bir katmandır.
        32: Bu katmandaki filtrelerin (kernellerin) sayısını belirtir. Her filtre, giriş görüntüsü üzerinden tarama yaparak farklı özellikleri öğrenir.
        kernel_size=(3, 3): Her bir filtrenin boyutunu belirtir (3x3 piksel boyutunda).
        activation="relu": Aktivasyon fonksiyonu olarak ReLU (Rectified Linear Unit) kullanılır, bu, negatif değerleri sıfırlar ve doğrusal olmayan bir fonksiyon ekler.
        input_shape=(width, height, 1): Giriş verisinin boyutlarını belirtir.
        (width, height) görüntü boyutlarını, 1 ise gri tonlamalı görüntü olduğunu belirtir (tek bir renk kanalı).

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu")):
        İkinci bir konvolüsyonel katman eklenir, bu sefer 64 filtre ile. Bu katman daha yüksek seviyeli özellikleri öğrenmek için kullanılır.

    model.add(MaxPooling2D(pool_size=(2, 2))):
        MaxPooling2D: Bu katman, havuzlama (pooling) işlemi gerçekleştirir ve her 2x2 bölgedeki en büyük değeri seçerek boyutları küçültür.
        Bu, modelin hesaplama maliyetini azaltırken önemli özellikleri korur.

    model.add(Dropout(0.25)):
        Dropout: Bu katman, belirtilen oranda (0.25, yani %25) rastgele nöronları devre dışı bırakır. Bu, modelin aşırı öğrenmesini (overfitting) önlemek için kullanılır.

    model.add(Flatten()):
        Flatten: Bu katman, çok boyutlu bir çıktıyı tek boyutlu bir vektöre düzleştirir. Bu, yoğun (dense) katmanlara geçiş yapmak için gereklidir.

    model.add(Dense(128, activation="relu")):
        Dense: Bu, tamamen bağlı (fully connected) bir katmandır. 128 nöron içerir ve ReLU aktivasyon fonksiyonu kullanır.
        Bu katman, öğrenilen özellikleri birleştirerek sınıflandırma işlemi için kullanılır.

    model.add(Dropout(0.4)):
        Bu sefer %40 oranında dropout uygulanır. Bu, son sınıflandırma katmanına geçişten önce aşırı öğrenmeyi önlemek için ek bir güvenlik sağlar.

    model.add(Dense(3, activation="softmax")):
        Son katman, 3 nöronlu bir dense katmandır (örneğin, 3 sınıfı temsil etmek için).
        softmax: Bu aktivasyon fonksiyonu, sınıflar arasında olasılık dağılımı üretir ve her sınıf için bir olasılık döndürür.
        Bu, çok sınıflı sınıflandırma problemleri için yaygın olarak kullanılır.

Özet

Bu kod bloğu, temel bir CNN modelini tanımlar. Model, görüntü verilerinde özellikleri öğrenir, bu özellikleri daha yüksek seviyeli temsillere dönüştürür ve
son olarak bu bilgileri sınıflandırma yapmak için kullanır.
ReLU aktivasyon fonksiyonları, dropout katmanları ve softmax çıktısı, modelin hem doğrusal olmayan özellikleri öğrenmesini hem de aşırı öğrenmeyi önlemesini sağlar.
"""

# model oluştu modeli compaile (derlemek) için gerekli olan parametrelerimiz

model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])
# los sayesinde veri setinden dolayı oluşan kaybımızı öğreniyoruz ve ona göre işlemler gerçekleştiriyoruz
"""
Bu satır, CNN modelinizi derlemek (compile) için kullanılır. Derleme işlemi, modelin eğitimi sırasında kullanılacak kayıp fonksiyonunu, optimizasyon algoritmasını ve değerlendirme metriklerini belirler. Bu parametrelerin her biri modelin nasıl eğitileceği ve performansının nasıl değerlendirileceği konusunda kritik bir rol oynar.
Kodun Açıklaması

    model.compile(...):
        Bu fonksiyon, modelin nasıl optimize edileceğini ve nasıl değerlendirileceğini belirler.

    loss="categorical_crossentropy":
        Kayıp Fonksiyonu: categorical_crossentropy çok sınıflı sınıflandırma problemleri için kullanılır.
        Bu kayıp fonksiyonu, modelin tahmin ettiği olasılık dağılımı ile gerçek etiketler arasındaki farkı ölçer. Categorical crossentropy, modelin sınıflandırma hatalarını minimize etmeye çalışır. Bu kayıp fonksiyonu, her sınıfın one-hot encoded olduğu durumlar için uygundur.

    optimizer="Adam":
        Optimizasyon Algoritması: Adam optimizatörü (Adaptive Moment Estimation), yaygın olarak kullanılan bir optimizasyon algoritmasıdır.
        Adam, öğrenme oranını her parametre için adaptif olarak ayarlar ve hem momentumu hem de adaptif öğrenme oranını birleştirir.
        Bu, özellikle büyük veri setlerinde ve karmaşık modellerde etkili bir şekilde çalışır.

    metrics=["accuracy"]:
        Değerlendirme Metrikleri: accuracy, modelin performansını değerlendirmek için kullanılan bir metriktir.
        Eğitim ve test verilerinde modelin doğru sınıflandırma yapma oranını ölçer. Bu metrik, modelin başarısını takip etmek için kullanılır.

Özet

Bu satır, modelin eğitim sürecinde nasıl optimize edileceğini ve performansının nasıl değerlendirileceğini belirler.
categorical_crossentropy kayıp fonksiyonu, çok sınıflı sınıflandırma problemlerinde kullanılır.
Adam optimizatörü, hızlı ve etkili bir optimizasyon sağlar. accuracy metriği, modelin sınıflandırma doğruluğunu izlemek için kullanılır.
Bu ayarlarla model, veri üzerinde eğitilmeye hazır hale gelir.
"""
model.fit(train_X,train_y,epochs=35,batch_size=64)
"""
Bu satır, modelinizi belirli bir sayıda epoch boyunca eğitmek için kullanılır. Model, eğitim verilerini kullanarak ağırlıklarını günceller ve her epoch sonunda modelin performansı değerlendirilir. Şimdi bu kodun ayrıntılarını açıklayalım:
Kodun Açıklaması

    model.fit(train_X, train_y, epochs=35, batch_size=64):
        model.fit(...): Bu fonksiyon, modeli belirli sayıda epoch boyunca eğitim verisi üzerinde eğitir.
        train_X giriş verilerini, train_y ise bu verilere karşılık gelen hedef etiketlerini içerir.

    train_X:
        Eğitim verisi olarak kullanılan özellikler (girdiler). Bu veri seti, modelin girdileri olarak kullanılır ve modelin öğrenmesi gereken bilgileri içerir.

    train_y:
        Eğitim verisi için hedef etiketler (çıktılar). Modelin tahmin etmeyi öğrenmesi gereken değerlerdir.

    epochs=35:
        Eğitim süresince model, veri seti üzerinde kaç kez eğitilecek (veya veri seti üzerinden kaç kez geçilecek).
        Bu durumda, model 35 epoch boyunca eğitilecektir. Her epoch, tüm eğitim verisinin bir kez modelden geçirilmesi anlamına gelir.

    batch_size=64:
        Her güncelleme için kullanılacak veri örneği sayısı. Yani model, 64 veri örneğini bir araya getirip, bunlar üzerinde ağırlıklarını günceller.
        Bu, bellek kullanımı ve eğitim süresi üzerinde doğrudan bir etkiye sahiptir.
        Küçük batch boyutları modelin daha sık güncellenmesini sağlarken, büyük batch boyutları daha dengeli güncellemeler yapar.
        """
score_train=model.evaluate(train_X,train_y)
print("Eğitim doğruluğu : %",score_train[1]*100)
score_test=model.evaluate(test_X,test_y)
print("Test doğruluğu : %",score_test[1]*100)

# şimdi elde ettiğimiz sonuçları kayıt işlemi
open("model_new.json","w").write(model.to_json())
model.save_weights("trex_weight.h5")