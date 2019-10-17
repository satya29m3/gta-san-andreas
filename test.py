import cv2
import numpy as np 
import matplotlib.pyplot as plt
from grabscreen import grab_screen
from directkeys import PressKey, ReleaseKey, W, A, S, D

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def roi(image):
    height = image.shape[0]
    polygons = np.array([
        [(10,500),(10,300),(300,200),(500,200),(800,300),(800,500)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked = cv2.bitwise_and(image,mask)
    return masked

def display_lines(image , lines ):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2  in lines:
            #x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image , (x1,y1),(x2,y2),(255,0,0),10)
    
    return line_image


def make_cord(image , line_parameters):
    slope ,intercept  = line_parameters
    y1  = image.shape[0]
    y2  = int(y1*(3/5))
    x1  = int((y1 - intercept)/slope)
    x2  = int((y2 - intercept)/slope)

    return np.array([x1,y1,x2,y2])


def averageslp(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameter = np.polyfit((x1,x2),(y1,y2),1)
        slope  =  parameter[0]
        intercept = parameter[1]
        if slope <0:
            left_fit.append((slope,intercept))
        else :
            right_fit.append((slope,intercept))
    left_fit_avg  = np.average(left_fit,axis  = 0)
    right_fit_avg  = np.average(right_fit,axis  = 0)
    left_line = make_cord(image , left_fit_avg)
    right_line = make_cord(image , right_fit_avg)
    return np.array([left_line,right_line])

def slope(lines):
    
    if lines is not None:
       x1,y1,x2,y2 = lines[0]
       x3,y3,x4,y4 = lines[1]
       slp = (((y2-y1)/(x2-x1)),((y4-y3)/(x4-x3)))
       print(slp)
    else:
        slp = (0.5,0.5)
    return slp


def preprocess(lane_image):
    canny1 = canny(lane_image)
    masked = roi(canny1)
    lines = cv2.HoughLinesP(masked,2,(np.pi/180),100,np.array([]),minLineLength = 29,maxLineGap = 5)
    try:
       average_line =  averageslp(lane_image,lines)
##       print (average_line.shape)
       slp = slope(average_line)
       line_image = display_lines(lane_image , average_line)
       crop  = cv2.addWeighted(lane_image,0.8, line_image,1,1)
       m1,m2 = slp
    except Exception as e:
        crop = lane_image
        m1,m2 = (0,0)
        print(str(e))
        pass

    return crop,m1,m2

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)

def right():
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def slow():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

while True:
    frame = grab_screen(region=(0,40,800,640))
    lane_image = np.copy(frame)
    crop ,m1,m2 = preprocess(lane_image)

    cv2.imshow('result',crop)
    if m1<0 and m2<0 :
        right()
    elif m1>0 and m2>0:
        left()
    else:
        straight()


    if cv2.waitKey(1) & 0xFF  == ord('q'):
        break

cv2.destroyAllWindows()

