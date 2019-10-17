import cv2
import numpy as np 
import matplotlib.pyplot as plt
from grabscreen import grab_screen
from directkeys import PressKey, ReleaseKey, W, A, S, D
from keras.models import load_model
import keras
from PIL import ImageGrab
from getkeys import key_check
import time
model = load_model('gta_model.h5')

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)

def main():
    last_time = time.time()

    for i in list(range(7))[::-1]:
        print(i+1)
        time.sleep(1)

    while(True):
        screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        print('loop took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        screen = cv2.cvtColor(screen , cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen,(224,224))
        cv2.imshow('',screen)
        moves = list(np.around(model.predict([screen.reshape(-1,224,224,1)])[0]))
        print('moves :',moves)

        if moves == [1,0,0]:
            left()
        elif moves == [0,1,0]:
            straight()
        elif moves == [0,0,1]:
            right()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)
main()
