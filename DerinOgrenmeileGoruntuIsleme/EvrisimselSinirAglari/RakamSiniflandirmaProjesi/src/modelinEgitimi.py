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

# verimizi inceledik baktık hepsi hemen hemen aynı boyutta e bu tanımayı azaltır bu yüzden zoom in-out yapacağız
path="../input/myData"
myList=os.listdir(path)
noOfClasses=len(myList)
print("Label sayısı : ",noOfClasses)


images=[]
classNo=[] # etiketimiz

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