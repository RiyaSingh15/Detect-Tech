from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import math
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


class Detect:
    def __init__(self):
        self.emotion_model = Sequential()
        self.emotion_model.add(Conv2D(32, kernel_size=(
            3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.emotion_model.add(
            Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Dropout(0.25))
        self.emotion_model.add(
            Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(
            Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
        self.emotion_model.add(Dropout(0.25))
        self.emotion_model.add(Flatten())
        self.emotion_model.add(Dense(1024, activation='relu'))
        self.emotion_model.add(Dropout(0.5))
        self.emotion_model.add(Dense(7, activation='softmax'))
        self.emotion_model.load_weights('model.h5')
        cv2.ocl.setUseOpenCL(False)
        self.emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ",
                             3: "   Happy   ", 4: "  Neutral  ", 5: "    Sad    ", 6: "Surprised"}
        self.emoji_dist = {0: "emojis/angry.png", 2: "emojis/disgusted.png", 2: "emojis/fearful.png",
                           3: "emojis/happy.png", 4: "emojis/neutral.png", 5: "/emojis/sad.png", 6: "emojis/surprised.png"}
        self.show_text = [0]
        self.root = Tk()
        self.gui()
        self.root.mainloop()

    def gui(self):
        self.inputVideo()
        self.outputVideo()
        self.root.resizable(0, 0)
        self.root.title("Detect Tech")
        heading = Label(self.root, text="Detect Tech",
                        font=("Arial bold", 100), padx=100)
        label1 = Label(self.root, text="Input", font=("Arial", 50))
        label2 = Label(self.root, text="Output", font=("Arial", 50))
        button1 = Button(self.root, text="Go Again", font=(
            "Arial bold", 30), fg='green', borderwidth=5, padx=20, command=self.gui)
        button2 = Button(self.root, text="Quit", font=(
            "Arial bold", 30), fg='red', borderwidth=5, padx=20, command=self.root.destroy)
        canvas1 = Label(self.root, height=300, width=300)
        canvas2 = Label(self.root, height=300, width=300)
        emotion_output = Label(
            self.root, text=self.emotion_dict[self.show_text[0]], font=("Arial", 15))
        gender_output = Label(
            self.root, text="Male/Female", font=("Arial", 15))
        age_output = Label(
            self.root, text="0-3/4-7/8-14/15-25/25-38/38-48/48-60/60-100", font=("Arial", 15))
        iimg = cv2.imread("Input.png")
        oimg = cv2.imread("Output.png")
        width = 290
        height = 290
        dim = (width, height)
        inp_resized = cv2.resize(iimg, dim, interpolation=cv2.INTER_AREA)
        inp_img = Image.fromarray(inp_resized)
        out_resized = cv2.resize(oimg, dim, interpolation=cv2.INTER_AREA)
        out_img = Image.fromarray(out_resized)
        self.inputtk = ImageTk.PhotoImage(image=inp_img)
        canvas1.configure(image=self.inputtk)
        self.outputtk = ImageTk.PhotoImage(image=out_img)
        canvas2.configure(image=self.outputtk)
        heading.grid(row=0, column=0, columnspan=3)
        label1.grid(row=1, column=0)
        label2.grid(row=1, column=2)
        canvas1.grid(row=2, column=0)
        canvas2.grid(row=2, column=2)
        emotion_output.grid(row=3, column=2)
        gender_output.grid(row=4, column=2)
        age_output.grid(row=5, column=2)
        button1.grid(row=6, column=0)
        button2.grid(row=6, column=2)

    def inputVideo(self):
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            self.root.destroy()
            print("Can't open the camera")
        i = 0
        while(i < 20):
            flag, frame = capture.read()
            i += 1
        cv2.imwrite("Input.png", frame)
        bounding_box = cv2.CascadeClassifier(
            'D:\\Softwares\\Anaconda\\envs\\tf\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = self.emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            self.show_text[0] = maxindex

    def outputVideo(self):
        frame1 = cv2.imread(self.emoji_dist[self.show_text[0]])
        pic = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        cv2.imwrite("Output.png", pic)


if __name__ == "__main__":
    obj = Detect()
