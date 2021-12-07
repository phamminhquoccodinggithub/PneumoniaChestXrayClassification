#dùng Tkinter tạo giao diện
from tkinter import*
main = Tk()
main.title("Pneumonia detection")
main.geometry("450x730")
main["bg"] = "Black"
main.attributes("-topmost", True) #allways in top
but1 = Button(main, text = "Insert Picture", font = ("Times New Roman",13))
but1.place (x = 130, y = 50)
but2 = Button(main, text = "Predict", font = ("Times New Roman",13))
but2.place (x = 240, y = 50)
T1 = Text(main, bg = "White", height = 15 , width = 50 )
T1.place (x = 25, y = 150)
lb1 = Label(main, text = "Picture:", bg = "Black", font = ("Times New Roman",13), fg = "White")
lb1.place (x = 25, y = 100)
lb2 = Label(main, text = "Predict Accuracy:", bg = "Black", font = ("Times New Roman",13), fg = "White")
lb2.place (x = 25, y = 420)
T2 = Text(main, bg = "White", height = 2 , width = 10 )
T2.place (x = 25, y = 470)
main.mainloop()