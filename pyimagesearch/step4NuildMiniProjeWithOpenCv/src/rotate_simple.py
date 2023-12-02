import numpy as np
import argparse
import imutils
import cv2
'''
argparse Modülü <br>
Giriş : <br>
Yazdığımız her uygulama grafik arayüzüne sahip olmaz. Bazı uygulamalar komut satırına daha uygundur ve bu uygulamalar bazı parametrelere ihtiyaç duyar. argparse modülü kullanıcıdan aldığımız parametreler için yardım mesajları, nasıl kullanıldığına yönelik mesajları üretir. Ayrıca bu modül kullanıcı geçersiz parametre girerse uygun hata mesajını bastırır.<br>
Bunu okuduğum site linki : https://python-istihza.yazbel.com/standart_moduller/argparse.html
'''
'''
# imutils : <br>
Yaya tanıma (pedestrian detection) video takip, kişi davranış analizi, akıllı arabalar için sürücü destek sistemi gibi farklı uygulamalarda kullanılabilir. Yaya tespiti probleminde makine öğrenmesi teknikleri yaygın olarak kullanılmaktadır, bu tekniklerde eğitilmiş sınıflandırıcılar ile görüntü içerisindeki yaya tespit edilmeye çalışılır. Sınıflama yapılırken yani bizim örneğimizde görüntüdeki insanları bulurken, uygun bir sınıflandırıcı bulmak kadar bulunacak nesneye ait olan seçici/ayırt edici bir özelliğin (feature) çıkarılabilir olması da çok önemlidir. Yayaların tanınması probleminde HOG (Histograms of Oriented Gradient) ile özellik çıkarılması yönteminin başarılı sonuçlar verdiği gösterilmiştir.<br>
HOG sadece kişi tanıma değil farklı nesnelerin tanınmasında da özellik çıkarımı için kullanılmaktadır ve OPENCV içerisinde implementasyonu gerçekleştirilmiştir. Bizim örneğimizde HOG kullanarak görüntü üzerindeki insanları tanımaya çalışacağız. Burada tahmin edeceğiniz üzere görüntü üzerinde bir pencere dolaştırmak ve bu pencerelerde arama yapmak gerekecek, Çünkü amacımız kişileri bulmak ve bir dikdörtgen içerisine almak. Görüntü üzerinde pencere gezdiriyoruz diyelim, kişileri neye göre ve nasıl tanıyacağız. Yani gezdirilen pencerenin içinde ağaç mı var, tabela mı, araba mı yoksa insan mı? Cevabını aradığımız soru bu!. İşte burada yazının başında bahsettiğimiz HOG iyi sonuçlar verdiği için tercih edilmekte. Tabi burada bir tanıma işlemi yapmak istediğimiz için, aslında bir makine öğrenmesi tekniği uygulamamız gerekiyor. Yani önce bir grup etiketlenmiş veriye HOG ile özellik çıkarımı yapıp, seçtiğimiz bir sınıflandırıcıya verip, bu sınıflandırıcıyı eğitmemiz gerekiyor. Daha sonra verdiğimiz yeni bir girdide, sınıflandırıcımız öğrenme kapasitesine göre bize bir cevap veriyor.

Neyseki OPENCV içerisinde halihazırda eğitilmiş HOG + Linear SVM modeli mevcut. Biz de bunu kullanarak görüntü ya da video üzerinde yaya tanıma işlemini gerçekleştirebiliriz. Bu yazı kapsamında vereceğim örnekte KAIST Multispectral Pedestrian Detection  veri seti kullanılmıştır Bu veri setinin avantajı RGB + Thermal band içermesidir. Hem böylelikle yaya tespitine örnek vermekle birlikte thermal bandının bu problemde kullanımının getirdiği avantajları da görmüş olacağız.<br>
Bunu okuduğum site linki : https://ibrahimdelibasoglu.blogspot.com/2017/01/python-yaya-tespiti-goruntu-isleme.html
'''

'''
 import argparse

def main():
    # Argüman analizi için bir ArgumentParser örneği oluşturun
    parser = argparse.ArgumentParser(description='Bu programın açıklaması.')

    # --input veya -i argümanı ekle
    parser.add_argument('--input', '-i', help='Giriş dosyasının yolu')

    # --output veya -o argümanı ekle
    parser.add_argument('--output', '-o', help='Çıkış dosyasının yolu')

    # --verbose veya -v argümanını ekleyerek daha fazla çıktı almayı etkinleştirin
    parser.add_argument('--verbose', '-v', action='store_true', help='Daha fazla çıktı al')

    # Argümanları analiz et
    args = parser.parse_args()

    # Elde edilen argümanları kullanarak işlemleri gerçekleştirin
    if args.verbose:
        print('Daha fazla çıktı etkinleştirildi.')

    if args.input:
        print(f'Giriş dosyası: {args.input}')

    if args.output:
        print(f'Çıkış dosyası: {args.output}')

if __name__ == '__main__':
    main()

 '''



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="\input")
args = vars(ap.parse_args())
# load the image from disk
image = cv2.imread(args["image"])
# loop over the rotation angles
for angle in np.arange(0, 360, 15):
	rotated = imutils.rotate(image, angle)
	cv2.imshow("Rotated (Problematic)", rotated)
	cv2.waitKey(0)
# loop over the rotation angles again, this time ensuring
# no part of the image is cut offS
for angle in np.arange(0, 360, 15):
	rotated = imutils.rotate_bound(image, angle)
	cv2.imshow("Rotated (Correct)", rotated)
	cv2.waitKey(0)
 
