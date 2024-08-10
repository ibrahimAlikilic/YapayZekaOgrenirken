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

print("a")
