import cv2
import numpy as np
import os

#########################################################

# elimde var olan sınıflarım
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
# her bir sınıf için renk belirleyelim
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#########################################################

# mobilNet SSD mizi içeri aktaralım
# elimizde olan hazır modeli kullandık
net = cv2.dnn.readNetFromCaffe("../input/SSD/MobileNetSSD_deploy.prototxt.txt", "../input/SSD/MobileNetSSD_deploy.caffemodel")

#########################################################

# Şu anda bulunduğunuz dizin
current_directory = os.path.dirname(os.path.abspath(__file__))

# Resimlerimin olduğu dizin
images_directory = os.path.join(current_directory, '..', 'input', 'images')

# Resim dosyalarını listeleme
files = [f for f in os.listdir(images_directory) if f.endswith('.jpg')]

img_path_list = [os.path.join(images_directory, f) for f in files]

#############################

# işleyelim
for i in img_path_list:
    image = cv2.imread(i)
    (h, w) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    # detections ları görselleştirelim
    for j in np.arange(0, detections.shape[2]):
        
        confidence = detections[0, 0, j, 2]
        
        if confidence > 0.30:
            
            idx = int(detections[0, 0, j, 1])
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            label = "{}: {:.2f}".format(CLASSES[idx], confidence)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 16 if startY - 16 > 15 else startY + 16
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
    cv2.imshow("ssd", image)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        continue

cv2.destroyAllWindows()
