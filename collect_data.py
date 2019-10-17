import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os

def keys_to_output(keys):
    output = [0,0,0]
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1 
    elif 'W' in keys:
        output[1] = 1
    return output


file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exist  , loading previous data')
    training_data = list(np.load(file_name))
else:
    print('creating new')
    training_data = []

def main():
    for i in list(range(4)):
        print(i+1)
        time.sleep(1)
    while(True):
         screen = grab_screen(region = (0,40,800,640))
         lasttime = time.time()
         screen = cv2.cvtColor(screen , cv2.COLOR_BGR2GRAY)
         screen  = cv2.resize(screen , (224,224))
         keys = key_check()
         output = keys_to_output(keys)
         training_data.append([screen,output])

         if cv2.waitKey(25) & 0xFF == ord('q'):
             cv2.destroyAllWindows()
             break
         if len(training_data)%500 ==0:
             print(len(training_data))
             np.save(file_name,training_data)


main()
