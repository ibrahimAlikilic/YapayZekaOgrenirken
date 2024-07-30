import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def Kayit(variables):
    # "output" klasörünün yolu
    output_dir = ''
    
    # Klasör yoksa oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Her bir değişkeni görselleştirip kaydetme işlemi
    for var_name, var_data in variables.items():
        # Değişkeni DataFrame'e dönüştürme (eğer değilse)
        if not isinstance(var_data, pd.DataFrame):
            var_data = pd.DataFrame(var_data)
    
        # DataFrame'i bir ısı haritası olarak görselleştirme
        plt.figure(figsize=(5, len(var_data) // 2))
        sns.heatmap(var_data, annot=True, cmap="coolwarm", cbar=False, annot_kws={"size": 10}, fmt="g") # bilimsel kaydetmeyi engelliyor
    
        # Görseli "output" klasörüne değişken adıyla bir dosyaya kaydetme
        file_path = os.path.join(output_dir, f'{var_name}.png')
        plt.savefig(file_path, bbox_inches='tight')
        plt.close() 
    
    print("Tüm değişkenler görselleştirilip dosyalara kaydedildi.")
