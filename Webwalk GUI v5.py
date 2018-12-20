from Tkinter import *
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style

kernel = np.ones((20,20),np.uint8)
cam = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX

global clt,cht,thr,wwid,awid,pgStr,valStr
window = Tk()
clt = DoubleVar()
cht = DoubleVar()
thr = DoubleVar()

wwid = StringVar()
awid = StringVar()
pgStr = StringVar()
valStr = StringVar()
def setTrue():
    global cbuttonState
    cbuttonState = True
    
def makeGui():

    global clt,cht,thr,wwid,awid,pgStr,valStr,hou
 
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
    b1 = Button(window,text = "Calibrate",command = setTrue)
    b1.grid(column=0, row=10)
    
    wwid.set(str(wWidth))
    awid.set(str(aWidth))

def filterImg(img,kernel,cannyLowThresh,cannyHighThresh):
    rows,cols = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(opening,cannyLowThresh,cannyHighThresh)
    return edges
def findCluster(vlines,num):
    emeans = KMeans(n_clusters = num).fit(vlines)
    entroids = emeans.cluster_centers_.astype(int)
    entroids.sort(0)
    return entroids
    
def findLines(img,thresh,orient):
    
    lines = cv2.HoughLinesP(img, rho=1, theta=np.pi / 180,
                            threshold=thresh, minLineLength=100, maxLineGap=20)
    #print(lines)
    #lines.resize(len(lines),4)
    #print(lines)
    vlines = np.ndarray(lines.shape, int)
    #print (vlines.shape)
    i = 0
    if orient:
        #print(lines)
        for line in lines:
            x1, y1, x2, y2, = line[0]
            if abs(y1 - y2) > abs(x1 - x2):
                #print(line[0])
                vlines[i] = x1, y1, x2, y2
                i = i + 0
    else:
        for line in lines:
            x1, y1, x2, y2, = line[0]
            if abs(y1 - y2) < abs(x1 - x2):
                vlines[i] = x1, y1, x2, y2
                i = i + 0
    vlines.resize(i,4)
    return vlines
                        
def printLines(img,lines, color):
    for line in lines:
        x1, y1, x2, y2, = line[0]
        cv2.line(img, (x1, y1), (x2,y2), (0,0,255),3)
    return img

def main():
    global clt,cht,thr,wwid,awid,pgStr,valStr,wWidth,aWidth
    cbuttonState = False
    cannyLowThresh = 20
    cannyHighThresh = 45
    lineThresh = 20
    wWidth = 60
    aWidth = 40
    disPerPix = 1.0
    pixGap = 0
    cbutton = False
    val = 0.0
    lastval = 0.0
    measure = 0.0
    color = (0,0,0)
    
    makeGui()
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
        try:
            
            edges = filterImg(img,kernel,cannyLowThresh,cannyHighThresh)
            
            
        except:
            print("No edges detected")
            #img = printLines(img, entroids, (255,255,0))
        try:
            vlines = findLines(edges,int(lineThresh),1)
            print(vlines)
        except:
            print("vlines not found")
        try:    
            entroids = findCluster(vlines,2)
            img = printLines(img, entroids, (255,255,0))
        except:
            print("entroids not found")
           
        

            
        cv2.imshow("Camera", img)
        cv2.imshow("Canny", edges)
        window.update()
if __name__ == '__main__':
    main()
    
    cv2.destroyAllWindows()
