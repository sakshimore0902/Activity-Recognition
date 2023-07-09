#import the necessary modules from the Tkinter library, which is used to create the graphical user interface (GUI) for the application.
from tkinter import *
from tkinter import Tk, ttk
import tkinter as tk
from tkinter import filedialog

#import various libraries used for deep learning, video processing, and data manipulation. load_model is imported from Keras to load a pre-trained model, deque is used to create a queue-like data structure, numpy is used for numerical computations, argparse is used to parse command-line arguments, pickle is used for object serialization, and cv2 is the OpenCV library for computer vision.
from keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2

#it creates tkinter root window
root = tk.Tk()
# sets the minimum size of the window to 500x500 pixels.
root.minsize(500, 500)
#it is global file to store the path of selected video
global filename
#loading the pre-trained model
model = load_model('model/activity.model')
lb = pickle.loads(open("model/lb.pickle", "rb").read())
#creates a NumPy array to store the mean pixel values used for preprocessing frames before feeding them into the model
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=128)

#browering the video file
def browseFile():
    global filename
    filename = filedialog.askopenfilename()
    entry_level.insert(0, filename)

#When called, it performs the video prediction algorithm. 
def runAlgo():

#is responsible for processing each frame of the video, making predictions, writing the frames with predictions to an output video file, and displaying the frames in a window.
    global filename
    vs = cv2.VideoCapture(filename)
    writer = None
    (W, H) = (None, None)
    while True:
        #reads the next frame
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)).astype("float32")
        frame -= mean
        preds = model.predict(np.expand_dims(frame, axis=0))[0]
        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = lb.classes_[i]
        text = "activity: {}".format(label)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.25, (0, 255, 0), 5)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter('output/lifting_128avg.avi', fourcc, 30,
                                     (W, H), True)
        writer.write(output)
        cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()


text1 = tk.Label(root, text='Video File', width=10)
text1.place(x=20, y=30)

entry_level = tk.Entry(root, width=30)
entry_level.place(x=100, y=30)

button1 = tk.Button(root, text='Browse', command=browseFile)
button1.place(x=320, y=30)


button2 = tk.Button(root, text='Predict', command=runAlgo)
button2.place(x=100, y=80)

#starts the Tkinter event loop, which handles user interactions with the GUI and keeps the application running until the main window is closed.
root.mainloop()
