import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2
train_data = np.load('train_dat.npy')
##train_data  = np.load('training_data.npy')
##train_data1 = np.load('training_data1.npy')
##train_data2 = np.load('training_data2.npy')
##train_data3 = np.load('training_data3.npy')
##train_data4 = np.load('training_data4.npy')
##print(len(train_data)+len(train_data1)+len(train_data2)+len(train_data3)+len(train_data4))
##
##left = []
##forward = []
##right = []
##
##for data in train_data:
##    img = data[0]
##    choice = data[1]
##    if (choice == [1,0,0]):
##        left.append([img,choice])
##    elif (choice == [0,1,0]):
##        forward.append([img,choice])
##    elif (choice == [0,0,1]):
##        right.append([img,choice])
##
##
##for data in train_data1:
##    img = data[0]
##    choice = data[1]
##    if (choice == [1,0,0]):
##        left.append([img,choice])
##    elif (choice == [0,1,0]):
##        forward.append([img,choice])
##    elif (choice == [0,0,1]):
##        right.append([img,choice])
##
##
##
##for data in train_data2:
##    img = data[0]
##    choice = data[1]
##    if (choice == [1,0,0]):
##        left.append([img,choice])
##    elif (choice == [0,1,0]):
##        forward.append([img,choice])
##    elif (choice == [0,0,1]):
##        right.append([img,choice])
##
##
##for data in train_data3:
##    img = data[0]
##    choice = data[1]
##    if (choice == [1,0,0]):
##        left.append([img,choice])
##    elif (choice == [0,1,0]):
##        forward.append([img,choice])
##    elif (choice == [0,0,1]):
##        right.append([img,choice])
##
##
##for data in train_data4:
##    img = data[0]
##    choice = data[1]
##    if (choice == [1,0,0]):
##        left.append([img,choice])
##    elif (choice == [0,1,0]):
##        forward.append([img,choice])
##    elif (choice == [0,0,1]):
##        right.append([img,choice])
##
##
##
##
##
##
##
##
##
##
##forward = forward[:len(left)][:len(right)]
##left = left[:len(forward)]
##right = right[:len(forward)]
##
##
##
##final_data = left+forward+right
##
##print(len(final_data))
##
##
##
##
##shuffle(final_data)
##np.save('train_dat.npy',final_data)
##
##
##
##









for data in train_data:
    img = data[0]
    choice= data[1]
    cv2.imshow('',img)
    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
