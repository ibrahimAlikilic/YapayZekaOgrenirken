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
    myImageList=os.listdir(path+"//"+str(i)) # i dedik çünkü dosya adlarım zaten 0-10