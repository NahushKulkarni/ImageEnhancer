import numpy as np
import cv2
import sys
import tkinter
from tkinter import filedialog
from PIL import Image
from model import resolve_single
from model.edsr import edsr
from model.wdsr import wdsr_b
from model.srgan import generator


def Load_Image(path):
    return np.array(Image.open(path))


def Save_Image(sr):
    sr=np.uint8(sr)
    cv2.imwrite(OutputFilename, sr)
    exit(0)


def Use_EDSR():
    model = edsr(scale=4, num_res_blocks=16)
    model.load_weights('weights/edsr-16-x4/weights.h5')
    lr = Load_Image(filename)
    sr = resolve_single(model, lr)
    Save_Image(sr)


def Use_WDSR():
    model = wdsr_b(scale=4, num_res_blocks=32)
    model.load_weights('weights/wdsr-b-32-x4/weights.h5')
    lr = Load_Image(filename)
    sr = resolve_single(model, lr)
    Save_Image(sr)


def Use_SRGAN():
    model = generator()
    model.load_weights('weights/srgan/gan_generator.h5')
    lr = Load_Image(filename)
    sr = resolve_single(model, lr)
    Save_Image(sr)


def GetFile():
    if len(sys.argv) > 1:
        return sys.argv[1], sys.argv[2]
    else:
        infile = filedialog.askopenfilename(title="Select Image")
        outfile = filedialog.asksaveasfilename(title="Select Output File")
        return infile, outfile


def Start():
    global filename, OutputFilename
    filename, OutputFilename = GetFile()
    if filename != "":
        return
    else:
        print("No File!")
        exit(0)

MainWindow = tkinter.Tk()
MainWindow.title("Image Super Resolution")
L1 = tkinter.Label(MainWindow, text="Select a method to use")
L1.pack()
B1 = tkinter.Button(MainWindow, text ="EDSR", command = Use_EDSR)
B1.pack()
B2 = tkinter.Button(MainWindow, text ="WDSR", command = Use_WDSR)
B2.pack()
B3 = tkinter.Button(MainWindow, text ="SRGAN", command = Use_SRGAN)
B3.pack()
filename, OutputFilename = [""] * 2
Start()
MainWindow.mainloop()
