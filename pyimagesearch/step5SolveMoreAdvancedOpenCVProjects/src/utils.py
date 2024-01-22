import numpy as np
import cv2
def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1) # Bu satır, kümeleme modeli tarafından atanan etiketlerin benzersiz sayısını (clt.labels_) alır ve bu sayıya kadar olan tamsayıları içeren bir dizi oluşturur. Bu dizinin her bir elemanı bir küme etiketini temsil eder.
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
	'''
	Bu satır, np.histogram fonksiyonunu kullanarak, kümeleme modeli tarafından atanan etiketlerin histogramını oluşturur. Histogram, her bir kümeye atanmış piksel sayısını gösterir.
    hist değişkeni, her bir etikete karşılık gelen piksel sayısını içeren bir dizi olur. _ değişkeni ise, histogramın bin sınırlarını içerir, ancak burada kullanılmaz.
	'''
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float") # Bu satır, histogram dizisini ondalık sayı türüne dönüştürür. Bu, sonraki işlemlerde normalizasyon yaparken daha hassas sonuçlar alabilmek için yapılır.
	hist /= hist.sum() # Bu satır, histogram dizisini normalize eder. Yani, toplam piksel sayısına bölerek, her bir kümeye düşen piksel oranlarını elde ederiz. Bu sayede, piksel sayılarına bağlı olarak farklı boyuttaki kümelerin karşılaştırılabilir hale gelmesini sağlarız.
	# return the histogram
	return hist
def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0 # Bu satır, çubuk grafikteki her bir renk bloğunun başlangıç noktasını belirlemek için kullanılan bir değişkeni başlatır.
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
		'''
		for (percent, color) in zip(hist, centroids):

    Bu satır, hist ve centroids dizilerini birlikte dolaşarak her bir kümenin yüzde oranını (percent) ve rengini (color) alır.

endX = startX + (percent * 300):

    Bu satır, her bir renk bloğunun genişliğini belirler. Yüzde oranı (percent) ile çubuk grafikteki genişlik arasındaki ilişki kullanılarak hesaplanır.

cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1):

    Bu satır, çubuk grafik üzerinde her bir renk bloğunu çizer. cv2.rectangle fonksiyonu, belirtilen başlangıç ve bitiş koordinatları arasında bir dikdörtgen çizer. color.astype("uint8").tolist() ifadesi, renk bilgisini uygun formata çevirir.
    -1 parametresi, dikdörtgenin içini doldurmak için kullanılır.

startX = endX:

    Bu satır, bir sonraki renk bloğunun başlangıç noktasını günceller. Böylece bir sonraki blok bir öncekinin bitiminden başlar.
	'''
	
	# return the bar chart
	return bar
