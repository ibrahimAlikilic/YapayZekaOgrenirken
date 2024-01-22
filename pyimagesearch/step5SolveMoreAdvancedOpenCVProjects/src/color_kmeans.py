'''
Take a second to look at the Jurassic Park movie poster above.

What are the dominant colors? (i.e. the colors that are represented most in the image)

Well, we see that the background is largely black. There is some red around the T-Rex. And there is some yellow surrounding the actual logo.

It’s pretty simple for the human mind to pick out these colors.

But what if we wanted to create an algorithm to automatically pull out these colors?

You might think that a color histogram is your best bet…

But there’s actually a more interesting algorithm we can apply — k-means clustering.

In this blog post I’ll show you how to use OpenCV, Python, and the k-means clustering algorithm to find the most dominant colors in an image.

OpenCV and Python versions:
This example will run on Python 2.7/Python 3.4+ and OpenCV 2.4.X/OpenCV 3.0+.
K-Means Clustering

So what exactly is k-means?

K-means is a clustering algorithm.

The goal is to partition n data points into k clusters. Each of the n data points will be assigned to a cluster with the nearest mean. The mean of each cluster is called its “centroid” or “center”.

Overall, applying k-means yields k separate clusters of the original n data points. Data points inside a particular cluster are considered to be “more similar” to each other than data points that belong to other clusters.

In our case, we will be clustering the pixel intensities of a RGB image. Given a MxN size image, we thus have MxN pixels, each consisting of three components: Red, Green, and Blue respectively.

We will treat these MxN pixels as our data points and cluster them using k-means.

Pixels that belong to a given cluster will be more similar in color than pixels belonging to a separate cluster.

One caveat of k-means is that we need to specify the number of clusters we want to generate ahead of time. There are algorithms that automatically select the optimal value of k, but these algorithms are outside the scope of this post.

'''
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())
# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# show our image
plt.figure() # Bu satır yeni bir figür oluşturur. Matplotlib'te bir figür, çizimlerinizin yapıldığı pencerenin veya sayfanın temsilcisidir.
plt.axis("off") # Bu satır, çizimdeki x ve y eksen etiketlerini ve işaretlerini kapatır. Burada kullanılma amacı, görseldeki resim üzerindeki x ve y eksen etiketlerini ve işaretlerini kaldırmaktır.
plt.imshow(image) # Bu satır, Matplotlib'in imshow fonksiyonunu kullanarak resmi gösterir. image değişkenini alır (bu, resmi temsil eden bir NumPy dizisidir) ve bunu figürde gösterir. Bu satır esasen resmi çizer.

image = image.reshape((image.shape[0] * image.shape[1], 3))
'''
Bu satır, image dizisini yeniden şekillendirir. reshape fonksiyonu, verilerini değiştirmeden dizinin şeklini değiştirmek için kullanılır. Burada, resmin orijinal dizisinin şekli (yükseklik, genişlik, 3) şeklinde olup, 3 renk kanalını (RGB) temsil eder.
reshape işlemi, 2D diziyi (resim) orijinal resmin her bir pikselini temsil eden bir satıra ve her bir pikselin RGB değerlerini içeren üç sütuna dönüştürür.
Yani bu satırdan sonra, image, orijinal resimdeki her bir pikseli temsil eden bir satıra sahip 2D bir dizi haline gelir ve her pikselin RGB değerlerini içeren üç sütunu vardır.
'''
clt = KMeans(n_clusters = args["clusters"]) # Bu satır, KMeans sınıfından bir kümeleme nesnesi oluşturur. n_clusters parametresi, belirtilen sayıda küme oluşturmak için kullanılır. Bu sayı, args sözlüğünden alınan "clusters" anahtarının değeridir. Yani, komut satırından girilen küme sayısıdır.
clt.fit(image)
'''
Bu satır, oluşturulan KMeans kümeleme nesnesini, önceki adımda hazırlanan düzleştirilmiş (reshape ile) image dizisi üzerine uygular. Yani, resimdeki pikselleri belirtilen küme sayısına göre kümelere ayırmak için eğitim işlemi gerçekleştirir.
Eğitim işlemi, KMeans algoritması tarafından gerçekleştirilen bir işlemdir. Algoritma, her bir pikseli, belirtilen küme sayısına göre en yakın küme merkezine atar ve bu merkezlere göre kümeleme işlemini gerçekleştirir.
clt.fit işleminden sonra, KMeans modeli, veri üzerinde eğitildi ve kümeleme işlemi gerçekleştirildi. Artık her bir piksel, bir küme numarası ile etiketlenmiş durumdadır.
'''
# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)
# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()