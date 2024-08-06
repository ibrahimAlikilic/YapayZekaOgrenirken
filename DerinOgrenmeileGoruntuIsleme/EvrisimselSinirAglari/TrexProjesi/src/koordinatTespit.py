import tkinter as tk
import pyautogui

def get_mouse_position():
    x, y = pyautogui.position() # fonksiyonu çağrılarak fare imlecinin anlık x ve y koordinatları alınır.
    position_str = f"X: {x} Y: {y}"
    label.config(text=position_str) # label etiketi güncellenir ve fare koordinatları ekranda gösterilir.
    root.after(100, get_mouse_position)  # Update every 100 milliseconds

root = tk.Tk() # kullanılarak ana pencere (root) oluşturulur.
root.title("Mouse Position Tracker")

label = tk.Label(root, font=('Helvetica', 14)) # label adında bir Label widget'ı (etiket) oluşturulur. Bu etiket, fare koordinatlarını gösterecek.
label.pack() # etiket pencereye eklenir.

get_mouse_position()

root.mainloop() # ifadesi ile ana döngü başlatılır ve pencere kapatılana kadar çalışmaya devam eder. Bu döngü, GUI uygulamalarının sürekli olarak ekranda kalmasını ve kullanıcı etkileşimlerini işleyebilmesini sağlar.
