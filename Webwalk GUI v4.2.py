from Tkinter import *
import cv2
import numpy as np
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from matplotlib import style

#cv2.namedWindow("test")

global cbuttonState
kernel = np.ones((20,20),np.uint8)
cam = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX

checkCalibrate = False

def calibrate(aW, wW, gap):
    if gap > 0:
        distance = float(abs(wW - aW) / 2.0)
        distPer = float(distance) / float(gap)
        print(distance)
        print(gap)
        print(distPer)
        checkCalibrate = True
        return distPer

def setTrue():
    global cbuttonState
    cbuttonState = True
    
def main():
    global cbuttonState
    kNum = 4
    cbuttonState = False
    gui = np.zeros((500,180), np.uint8)
    cannyLowThresh = 20
    cannyHighThresh = 45
    lineThresh = 20
    wWidth = 60
    aWidth = 40
    disPerPix = 1.0
    pixGap = 0
    window = Tk()
    window.title("Welcome to LikeGeeks app")
    #window.geometry('100x500')
    clt = DoubleVar()
    cht = DoubleVar()
    thr = DoubleVar()
    wwid = StringVar()
    awid = StringVar()
    pgStr = StringVar()
    valStr = StringVar()
    lnm = StringVar()
    cbutton = False
    val = 0.0
    lastval = 0.0
    measure = 0.0
    color = (0,0,0)
    
    lnm.set(kNum)
    
    lbl1 = Label(window, text="Canny Low")
    lbl1.grid(column=0, row=0)
    clow = Scale( window, variable = clt, orient = HORIZONTAL )
    clow.grid(column=0, row=1)
    lbl2 = Label(window, text="Canny High")
    lbl2.grid(column=0, row=2)
    chigh = Scale( window, variable = cht, orient = HORIZONTAL )
    chigh.grid(column=0, row=3)
    lbl3 = Label(window, text="Hough Thresh")
    lbl3.grid(column=0, row=4)
    hou = Scale( window, variable = thr, orient = HORIZONTAL )
    hou.grid(column=0, row=5)
    lbl4 = Label(window, text="Web Width")
    lbl4.grid(column=0, row=6)
    e1 = Entry(window,textvariable = wwid)
    e1.grid(column=0, row=7)
    lbl5 = Label(window, text="Antenna Width")
    lbl5.grid(column=0, row=8)
    e2 = Entry(window,textvariable = awid)
    e2.grid(column=0, row=9)
    lbl6 = Label(window, text="Number of Lines")
    lbl6.grid(column=0, row=10)
    e3 = Entry(window,textvariable = lnm)
    e3.grid(column=0, row=11)
    b1 = Button(window,text = "Calibrate",command = setTrue)
    b1.grid(column=0, row=12)
    
    wwid.set(str(wWidth))
    awid.set(str(aWidth))
    redCheck = False
    redLastCheck = False
    redCount = 0
    while True:

        if cannyHighThresh < cannyLowThresh:
            cannyHighThresh = cannyLowThresh
        
        clt.set(cannyLowThresh)
        cht.set(cannyHighThresh)        
        hou.set(lineThresh)
        
        if cv2.waitKey(20) == 27:
            window.destroy()
            break

        ret, frame = cam.read()
        img = frame

        rows,cols = img.shape[:2]
        M = np.float32([[1,0,100],[0,1,50]])
        img = cv2.resize(img,None,fx=1,fy=1,interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        #opening = cv2.bilateralFilter(gray, 11, 20, 20)
        check = True
        try:
            edges = cv2.Canny(opening,cannyLowThresh,cannyHighThresh)
            
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                  threshold=int(lineThresh), minLineLength=100, maxLineGap=20)
            lines.shape 
        except:
            check = False 
        if check:
            vlines = np.ndarray(lines.shape, int)
            for line in vlines:
                line[0] = 0,0,0,0
            i = 0
            for line in lines:
                x1, y1, x2, y2, = line[0]
                if abs(y1 - y2) > abs(x1 - x2):
                    vlines[i,0] = line[0]
                    i = 1 + i
            vlines.resize(i,1,4)
            try:
                vlines.shape
            except:
                check = False
            if check:
                flines = np.ndarray((i,4), int)
                maxlines = np.ndarray((i,1), int)
                i = 0
                for line in vlines:
                    x1, y1, x2, y2, = line[0]
                    maxlines[i] = np.sqrt(abs(x1 - x2)^2 + abs(y1 - y2)^2)
                    i = i + 1
                std = maxlines.std()
                avg = maxlines.mean()
                j = 0
                i = 0
                
                for line in vlines:
                    x1, y1, x2, y2, = line[0]
                    
                    if maxlines[i] >= (avg):
                        flines[j] = line[0]
                        j = j + 1
                        cv2.line(img, (x1, y1), (x2,y2), (0,0,255),3)
                    i = i+1

                vlines.resize(i,4)
                try:
                    emeans = KMeans(n_clusters = kNum ).fit(flines)
                    entroids = emeans.cluster_centers_.astype(int)
                    entroids.sort(0)
                except:
                    check = False
                if check:
                    check = False
                    checkOverlap = False
                    line2 = True
                    
                    for line in flines:
                        x1, y1, x2, y2, = line
                        for ent in entroids:
                            e0 , e1, e2, e3 = ent
                            if 15 >= abs(e0 - x1) or 15 >= abs(e2 - x1):
                                checkOverlap = True
                    for x in range(kNum):
                        if x == 0:
                            None
                        elif 15 >= abs(entroids[x,0]- entroids[(x-1),0]) or 15 >= abs(entroids[x,2]- entroids[(x-1),2]):
                            line2 = False
                        for y in range(4):
                            if  entroids[x,y] > 100000 or 0 >= entroids[x,y]:
                                line2 = False
                    if checkOverlap and line2:
                        check = True
                    if check:
                        sum1 = 0
                        print(entroids)
                        for line in entroids:
                            x1, y1, x2, y2, = line
                            cv2.line(img, (x1, y1), (x2,y2), (0,255,0),3)
                        sum1 = sum1 + abs(entroids[0,0] - entroids[(kNum - 1),0])
                        sum1 = sum1 + abs(entroids[0,2] - entroids[(kNum - 1),2])
                        sum1 = sum1 / 2
                        pixGap = sum1
                            
        if cbuttonState:
            disPerPix = calibrate(wWidth,aWidth,pixGap)
            wwid.set(str(wWidth))
            awid.set(str(aWidth))
            
            try: val = float(pixGap * disPerPix)
            except:
                val = 0
            measure = val 
        lastval = val
        
        try: val = float(pixGap * disPerPix)
        except:
            val = 0
        if abs(val - measure) > .5 and abs(lastval - measure) > .5:
            color = (0,0,255)
            redCheck = True
        else:
            color = (0,255,0)
            redCheck = False
        cv2.putText(img,str(pixGap),(10,30), font, 1,(255,255,0),2,cv2.LINE_AA)
        cv2.putText(img,str("%.2f" %val),(100,30), font, 1, color, 2,cv2.LINE_AA)
        cv2.putText(img,str(redCount),(300,30), font, 1,(0,0,255),2,cv2.LINE_AA)
        if redCheck and not redLastCheck:
            redCount = redCount + 1
        redLastCheck = redCheck
      
        cv2.imshow("Camera", img)
        cv2.imshow("Canny", edges)
    
        cbuttonState = False
        window.update()
        
        if unicode(lnm.get()).isnumeric():
            kNum = int(lnm.get())
        elif str(lnm.get()):
            lnm.set(str(kNum))

        
        if unicode(wwid.get()).isnumeric():
            wWidth = int(wwid.get())
        elif str(wwid.get()):
            wwid.set(str(wWidth))

        
        if unicode(awid.get()).isnumeric():
            aWidth = int(awid.get())
        elif str(awid.get()):
            awid.set(str(aWidth))
        cannyLowThresh = clt.get()
        cannyHighThresh = cht.get()
        lineThresh = hou.get()
if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
    
