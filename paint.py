from tkinter import colorchooser, messagebox
import tkinter as tk
from PIL import Image, ImageDraw
from random import randint
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import math
from scipy.ndimage.measurements import center_of_mass


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Mnist_paint')
        self.root.geometry('700x500')
        self.root.resizable(0, 0)

        self.brush_size = tk.IntVar(value=10)
        self.color = tk.StringVar(value='black')
        self.color_canvas = tk.StringVar(value='white')

        self.canvas = tk.Canvas(root, bg='white', width=400, height=400)
        self.canvas.grid(row=2, column=0, columnspan=7, padx=5, pady=5)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<Button-3>', self.popup)

        self.menu = tk.Menu(tearoff=0)

        self.image1 = Image.new('RGB', (400, 400), 'white')
        self.draw_img = ImageDraw.Draw(self.image1)

        try:
            self.model = load_model('mnist_model.keras')
        except Exception as e:
            messagebox.showerror("Ошибка", f"Невозможно загрузить модель: {e}")
            self.root.destroy()
            return

        self.init_ui()

    def init_ui(self):
        tk.Label(self.root, text='Параметры: ').grid(row=0, column=0, padx=6)

        tk.Button(self.root, text='Выбрать цвет кисти', width=15,
                  command=self.choose_color).grid(row=0, column=1, padx=6)
        tk.Button(self.root, text='Выбрать цвет фона', width=15,
                  command=self.choose_color_canvas).grid(row=1, column=1, padx=6)

        self.color_lab = tk.Label(self.root, bg=self.color.get(), width=10)
        self.color_lab.grid(row=0, column=2, padx=6)

        self.color_canvas_lab = tk.Label(
            self.root, bg=self.color_canvas.get(), width=10)
        self.color_canvas_lab.grid(row=1, column=2, padx=6)

        tk.Scale(self.root, variable=self.brush_size, from_=2, to=5,
                 command=self.select, orient=tk.HORIZONTAL, length=150).grid(row=0, column=3, padx=6)

        tk.Button(self.root, text='Очистить', width=10,
                  command=self.clear_canvas).grid(row=1, column=0)

        tk.Button(self.root, text='Сохранить рисунок', width=16,
                  command=self.save_image).grid(row=1, column=3)

        tk.Button(self.root, text='Определить цифру', width=15,
                  command=self.predict_digit).grid(row=1, column=4)

        self.prediction_label = tk.Label(
            self.root, text="Предсказанная цифра: ")
        self.prediction_label.grid(row=0, column=4, padx=6)

    def draw(self, event):
        brush_size = self.brush_size.get()
        x1, y1 = (event.x - brush_size), (event.y - brush_size)
        x2, y2 = (event.x + brush_size), (event.y + brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.color.get(), width=0)
        self.draw_img.ellipse((x1, y1, x2, y2), fill=self.color.get(), width=0)

    def choose_color(self):
        (rbx, hx) = colorchooser.askcolor()
        if hx:
            self.color.set(hx)
            self.color_lab['bg'] = hx

    def choose_color_canvas(self):
        (rbx, hx) = colorchooser.askcolor()
        if hx:
            self.canvas.configure(bg=hx)
            self.color_canvas_lab['bg'] = hx

    def select(self, value):
        self.brush_size.set(int(value))

    def getBestShift(self, img):
        cy, cx = center_of_mass(img)
        rows, cols = img.shape
        shiftx = np.round(cols/2.0 - cx).astype(int)
        shifty = np.round(rows/2.0 - cy).astype(int)
        return shiftx, shifty

    def shift(self, img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted

    def rec_digit(self, img_path):
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            gray = 255 - img
            (thresh, gray) = cv2.threshold(
                gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            while np.sum(gray[0]) == 0:
                gray = gray[1:]
            while np.sum(gray[:, 0]) == 0:
                gray = np.delete(gray, 0, 1)
            while np.sum(gray[-1]) == 0:
                gray = gray[:-1]
            while np.sum(gray[:, -1]) == 0:
                gray = np.delete(gray, -1, 1)
            rows, cols = gray.shape

            if rows > cols:
                factor = 20.0 / rows
                rows = 20
                cols = int(round(cols * factor))
                gray = cv2.resize(gray, (cols, rows))
            else:
                factor = 20.0 / cols
                cols = 20
                rows = int(round(rows * factor))
                gray = cv2.resize(gray, (cols, rows))

            colsPadding = (int(math.ceil((28 - cols) / 2.0)),
                           int(math.floor((28 - cols) / 2.0)))
            rowsPadding = (int(math.ceil((28 - rows) / 2.0)),
                           int(math.floor((28 - rows) / 2.0)))
            gray = np.lib.pad(gray, (rowsPadding, colsPadding), 'constant')

            shiftx, shifty = self.getBestShift(gray)
            shifted = self.shift(gray, shiftx, shifty)
            gray = shifted

            cv2.imwrite('temp_gray.png', gray)
            img = gray / 255.0
            img = np.array(img).reshape(-1, 28, 28, 1)
            out = str(np.argmax(self.model.predict(img)))
            os.remove('temp_gray.png')
            return out
        except Exception as e:
            messagebox.showerror(
                "Ошибка", f"Ошибка при определении цифры: {e}")
            return None

    def predict_digit(self):
        filename = 'temp_image.png'
        self.image1.save(filename)
        predicted_digit = self.rec_digit(filename)
        if predicted_digit is not None:
            self.prediction_label.config(
                text=f"Предсказанная цифра: {predicted_digit}", fg="red")
        os.remove(filename)

    def clear_canvas(self):
        self.canvas.delete('all')
        self.canvas['bg'] = 'white'
        self.draw_img.rectangle((0, 0, 400, 400), fill='white')
        self.prediction_label.config(text="Предсказанная цифра: ", fg="black")
        self.color_lab['bg'] = 'black'
        self.color_canvas_lab['bg'] = 'white'

    def save_image(self):
        filename = f'image_{randint(0, 1000)}.png'
        self.image1.save(filename)
        messagebox.showinfo(
            'Сохранение', f'Сохранено под названием {filename}')

    def popup(self, event):
        self.menu.post(event.x_root, event.y_root)


root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
