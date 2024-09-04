import os

# Fotoğrafların bulunduğu dizinin yolu
directory = "../../../../Masaüstü/DateSets/boston dynamics spot"

# Dizin içindeki dosyaları al ve yalnızca .jpg uzantılı olanları filtrele
files = [f for f in os.listdir(directory)]

# Dosyaları sıralı şekilde yeniden adlandır
for i, filename in enumerate(sorted(files), 1):
    old_file_path = os.path.join(directory, filename)
    new_file_name = f"{i}.jpg"
    new_file_path = os.path.join(directory, new_file_name)
    
    # Dosyayı yeniden adlandır
    os.rename(old_file_path, new_file_path)

print("Dosyalar başarıyla yeniden adlandırıldı!")
