import cv2
import numpy as np
import random
import requests

##########################################

url="http://192.168.1.104:8080/shot.jpg" # cap yerine bu var

birKereRenk=True # nesnelere renk atamasını 1 kere gerçekleştirmesi için

while True:
    
    ##########################################
    
    # telden alınan görüntünün düzenlemesi
    img_resp=requests.get(url)
    img_arr=np.array(bytearray(img_resp.content),dtype=np.uint8)
    img=cv2.imdecode(img_arr,cv2.IMREAD_COLOR)
    frame=cv2.resize(img,(640,840))    
   
    ##########################################
    
    frame_width=frame.shape[1]
    frame_height=frame.shape[0]

    ##########################################
    
    frame_blob=cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True,crop=False)
                                # resim , yolonun yazarlarının kabulü = 1/255 , indirmiş olduğumuz modele göre remimizin boyutunu resize2liyoruz , BGT2RGB , crop=kırpma

    ##########################################

    # labels.txt dosyasını oku
    path="../input/"
    with open(path+'labels.txt', 'r') as file:
        data = file.read()

    # Gereksiz karakterleri temizle ve listeye çevir
    labels = data.replace('"', '').split(',')
    if birKereRenk:
        colors = np.random.uniform(0, 255, size=(len(labels), 3))
        colors = colors.astype(int)
        birKereRenk=False
    
    #####################
    
    # Model variable
    model=cv2.dnn.readNetFromDarknet("../../../../Masaüstü/YOLOlar/YOLOv3/pretrained_model/yolov3.cfg","../../../../Masaüstü/YOLOlar/YOLOv3/pretrained_model/yolov3.weights")
    layers = model.getLayerNames() # layer = katman
    output_layert=[layers[layer-1] for layer in model.getUnconnectedOutLayers()]
    model.setInput(frame_blob)
    detection_layers=model.forward(output_layert)

    ##########################################
    
    # Non Max Ssuppression

    # Operation-1
    ids_list=[]
    boxes_list=[]
    confidences_list=[]
    # end of Operation-1

    ##########################################

    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            # güven skorü ile ilgilenelim
            score=object_detection[5:] # ilk 5 değer box ile ilgili
            # score içerisindeki en büyük değer benim predicted id olacak = tahmin edilen nesnemin index i
            predicted_id=np.argmax(score)
            confidence=score[predicted_id] # confidence = güven skoru

            # box çizim aşamasına geldik
            if confidence > 0.3:
                label=labels[predicted_id] # labeli ne onu alalım
                bounding_box=object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height])
                (box_center_x,box_center_y,box_width,box_height)=bounding_box.astype("int")
                start_x=int(box_center_x-(box_width/2))
                start_y=int(box_center_y-(box_height/2))

                # Non Max Ssuppression - Operation 2
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x,start_y,int(box_width),int(box_height)])
                # end of Operation-2
    # Non Max Ssuppression - Operation 3
    max_ids=cv2.dnn.NMSBoxes(boxes_list,confidences_list,0.5,0.4) # bu fonksiyon bana en yüksek güvenilirliğie sahip boxes ların id lerini döndürüyor
                            # son 2 parametre boundig_box ın güven skoru ve threshold değeri  (0.5,04 = bınlar optimal değerlerdir)
    for max_id in max_ids:
        if isinstance(max_id, (list, np.ndarray)):
                max_class_id = max_id[0]  # Eğer max_id bir liste ya da NumPy dizisi ise
                # max_id[0] dememizin sebebi max_id bir liste ve ben bunun 0. indexdeki değerine erişmek istiyorum
                # max_ids, bu id'leri içeren bir liste olarak döner. Bu listenin her bir elemanı, tek bir id'yi içeren başka bir liste olur. 
                # Bu nedenle, döngü içinde max_id bir liste olur, ve max_id[0] ifadesi bu listenin ilk (ve tek) elemanına, yani asıl id'ye erişmek için kullanılır.
        else:
            max_class_id = max_id  # Eğer max_id sadece bir sayı ise

        box=boxes_list[max_class_id]

        start_x=box[0]
        start_y=box[1]
        box_width=box[2]
        box_height=box[3]

        predicted_id=ids_list[max_class_id]
        label=labels[predicted_id]
        confidence=confidences_list[max_class_id]

    # end of Operation-3

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color=colors[predicted_id]
        box_color=[int(each) for each in box_color]# liste şeklinde olması gerekemkte # aslında yukarıda int olarak oluşturmuştum ama burada int olmayınca hata aldım
        cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),box_color,1)

        # label imde confidence değerimi de görmek istiyorum
        label="{}: {:.2f}%".format(label,confidence*100) # Hatırlatma : ".2f" . dan sonra 2 basamak olsun manasında
        print(f"prediction object {label}") # uçbirimde de görmek istedim
        cv2.putText(frame,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
    # Büyük görmek istiyorum
    # frame=cv2.resize(frame,(1280,960))
    cv2.imshow("Tespit (Detection)",frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
cv2.destroyAllWindows()