import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../input')))
from yolo_model import YOLO 

#########################################################

# YOLO modelini yükleyelim
yolo = YOLO(0.3, 0.3)

#########################################################

# Etiketleri yükleyelim
file = "../input/data/data/coco_classes.txt"
with open(file) as f:
    class_names = f.readlines()
all_classes = [c.strip() for c in class_names]

# Ara Kayıt
output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))
os.makedirs(output_directory, exist_ok=True)

class_names_path = os.path.join(output_directory, 'class_names.txt')
all_classes_path = os.path.join(output_directory, 'all_classes.txt')

if not os.path.exists(class_names_path):
    with open(class_names_path, 'w') as f:
        f.writelines(class_names)
    print(f"class_names {class_names_path} dizinine kaydedildi")
else:
    print(f"class_names.txt zaten mevcut, işlem yapılmadı.")

if not os.path.exists(all_classes_path):
    with open(all_classes_path, 'w') as f:
        for item in all_classes:
            f.write("%s\n" % item)
    print(f"all_classes {all_classes_path} dizinine kaydedildi")
else:
    print(f"all_classes.txt zaten mevcut, işlem yapılmadı.")

#########################################################

# VideoCapture nesnesi oluşturuluyor
cap = cv2.VideoCapture(0)
# Kare sayacı
frame_count = 0

#########################################################

# Predict aşaması
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret is None:
        break

    # Belirli bir kare sayısında bir işlem yapalım çünkü diğer türlü ok yavaş
    if frame_count % 5 == 0:  # 5 karede bir tahmin yapılacak
        pframe = cv2.resize(frame, (416, 416))
        pframe = np.array(pframe, dtype="float32")
        pframe /= 255.0
        pframe = np.expand_dims(pframe, axis=0)

        # Tahmin yap ve sonuçları kontrol et
        results = yolo.predict(pframe, pframe.shape)
        boxes, classes, scores = results
        print(f"classes : {classes}")
        print(f"score : {scores}")
        if results is not None:
            if boxes is not None and classes is not None and scores is not None:
                for box, cl, sco in zip(boxes, classes, scores):
                    x, y, w, h = box
                    left = max(0, min(np.floor(x + 0.5).astype(int), frame.shape[1] - 1))
                    top = max(0, min(np.floor(y + 0.5).astype(int), frame.shape[0] - 1))
                    right = max(0, min(np.floor(x + w + 0.5).astype(int), frame.shape[1] - 1))
                    bottom = max(0, min(np.floor(y + h + 0.5).astype(int), frame.shape[0] - 1))
                    print(f"Left: {left}, Top: {top}, Right: {right}, Bottom: {bottom}")
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 7)
                    cv2.putText(frame, "{}: {:.2f}".format(all_classes[cl], sco), 
                                (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                (0, 0, 255), 2, cv2.LINE_AA)
            else:
                print("others none")
        else:
            print("result=none")

    cv2.imshow("frame", frame)
    frame_count += 1

    if cv2.waitKey(2) & 0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
