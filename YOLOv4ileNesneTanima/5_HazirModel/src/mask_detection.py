# Kutuphaneler
import cv2
import numpy as np
import random

##########################################
# Resim
img=cv2.imread("../input/mask.jpg") # Hatırlatma : Resimleri BGR okur
img_width=640
img_height=427
img=cv2.resize(img,(img_width,img_height))
print(f"img shape : {img.shape}") # hatırlatma : height , weidth olarak verir

##########################################
# resmimizi blob formata çevirmemiz lazım ( resmin 4 boyutlu tensorlere çevrilmiş hali )


img_blob=cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)
                              # resim , yolonun yazarlarının kabulü = 1/255 , indirmiş olduğumuz modele göre remimizin boyutunu resize2liyoruz , BGT2RGB , crop=kırpma
print(f"img_blob shape : {img_blob.shape}") # hatırlatma : height , weidth olarak verir

##########################################
# Değişkenleri (variable) oluşturalım

labels = ["good","bad"]

colors = ["255,0,0","0,0,255"]
colors=[np.array(color.split(",")).astype("int") for color in colors]
colors=np.array(colors)
colors=np.tile(colors,(18,1))

#####################
# Model variable
model=cv2.dnn.readNetFromDarknet("../../../../Masaüstü/YOLOlar/Mask_Detection/yolov3_mask.cfg","../../../../Masaüstü/YOLOlar/Mask_Detection/yolov3_mask_last.weights")
layers = model.getLayerNames() # layer = katman

# layers tğm katmanları içeriyor benim istediğim şey detection (tespit) ( yani çıktı ) katmanları
# bunun için özel bir fonksiyon var ama onda şöyle bir durum var bze kaçıncı sırada olduğunu döndürüyor ama biz index olarak kullanıyoruz o yüzden -1 diyeceğiz
print("****************************")
print(f"model.getUnconnectedOutLayers() :  {model.getUnconnectedOutLayers()}")
output_layert=[layers[layer-1] for layer in model.getUnconnectedOutLayers()]
# Eğitim videosunda layer[0]-1 diyordu ama hata aldım ve benim mantığıma da uymamıştı yanlış olduğunu düşünüyordum ki hata aldım o yüzden bu şekilde düzenledim

model.setInput(img_blob)
# setInput: Bu yöntem, modele bir giriş (input) verisi sağlar. Yani, modelin işlem yapması için gerekli olan veriyi, bu fonksiyon aracılığıyla modele iletmiş oluyorsun.
detection_layers=model.forward(output_layert)
# forward() fonksiyonu, modelin ileri besleme işlemini gerçekleştirir. Bu, modelin giriş verisini (görüntüyü) işleyip çıktıları (nesne tespitlerini) üretmesi anlamına gelir.
# Bu fonksiyon, modelin belirtilen çıkış katmanlarının çıktısını döndürür.

##########################################

# Non Max Ssuppression

# Operation-1
ids_list=[]
boxes_list=[]
confidences_list=[]
# end of Operation-1

##########################################

print("****************************")
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
            bounding_box=object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
            # object_detection[0:4] ün döndürdüğü değerler normalize edilmiş durumdadır bu sebepten dolayı " * np.array([img_width,img_height,img_width,img_height])" yaıyoruz
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
    cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,1)
    
    # label imde confidence değerimi de görmek istiyorum
    label="{}: {:.2f}%".format(label,confidence*100) # Hatırlatma : ".2f" . dan sonra 2 basamak olsun manasında
    print(f"prediction object {label}") # uçbirimde de görmek istedim
    cv2.putText(img,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
# Büyük görmek istiyorum
img=cv2.resize(img,(1280,960))
cv2.imshow("Tespit (Detection)",img)
cv2.waitKey(0)
cv2.destroyAllWindows()