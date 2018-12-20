import cv2
import numpy as np
from sklearn.cluster import KMeans
import cvui
import matplotlib.pyplot as plt
from matplotlib import style
WINDOW_NAME = 'CVUI GUI'
#cv2.namedWindow("test")
cannyLowThresh = [20]
cannyHighThresh = [45]
lineThresh = [20]
wWidth = [60]
aWidth = [40]

kernel = np.ones((20,20),np.uint8)
cam = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_SIMPLEX




def calibrate(aW, wW, gap):
    if gap > 0:
        distance = abs(wW - aW) / 2
        distPer = float(distance) / float(gap)
        print(distance)
        print(gap)
        print(distPer)
        return distPer

def main():
    gui = np.zeros((500,180), np.uint8)
    cvui.init(WINDOW_NAME)
    disPerPix = 1.0
    pixGap = 0
    while True:
       
        cvui.window(gui, 0, 0, 180, 500, 'Settings')	
        # Two trackbars to control the low and high threshold values
        # for the Canny edge algorithm.
        if cannyHighThresh[0] < cannyLowThresh[0]:
            cannyHighThresh[0] = cannyLowThresh[0]
        cvui.text(gui, 10, 30,  'Canny Low')
        cvui.trackbar(gui, 0, 50, 165, cannyLowThresh, 1, 300, 1,"%.0Lf")
        cvui.text(gui, 10, 100,  'Canny High')
        cvui.trackbar(gui, 0, 120, 165, cannyHighThresh, 1, 300, 1,"%.0Lf")
        cvui.text(gui, 10, 170,  'Hough Threshold')
        cvui.trackbar(gui, 0, 190, 165, lineThresh, 1, 300 , 1,"%.0Lf" )
        
        cvui.text(gui, 10, 250,  'Web Width')
        
    
        cvui.trackbar(gui, 0, 260, 165, wWidth, 10, 100 , 1, "%.0Lf")
        cvui.text(gui, 10, 320,  'Antenne Width')
        cvui.trackbar(gui, 0, 340, 165, aWidth, 10, 100 , 1, "%.0Lf" )
        
        if cv2.waitKey(20) == 27:
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
            edges = cv2.Canny(opening,cannyLowThresh[0],cannyHighThresh[0])
            
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                                  threshold=int(lineThresh[0]), minLineLength=100, maxLineGap=20)
            lines.shape    
            
        except:
            check = False
        #print(check)   
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
                blines = np.ndarray((i,2), int)
                elines = np.ndarray((i,2), int)
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
                    
                    if maxlines[i] >= (avg + (0 * std))*0:
                        #print(x1)
                        flines[j] = line[0]
                        j = j + 1
                        cv2.line(img, (x1, y1), (x2,y2), (0,0,255),3)
                        #print(maxlines[i])
                    i = i+1

                vlines.resize(i,4)
                #print(vlines)
                blines.resize(j,2)
                elines.resize(j,2)
                #print(blines)
                
                if blines.any() and elines.any():

                    try:
                        #print(vlines[0,2:4])
                        emeans = KMeans(n_clusters=2).fit(flines)
                        entroids = emeans.cluster_centers_.astype(int)
                        entroids.sort(0)
                        #bmeans = KMeans(n_clusters=2).fit(flines[2:4])
                    except:
                        check = False
                    check = False
                    line1 = False
                    line2 = False
                    for line in flines:
                        x1, y1, x2, y2, = line
                        if 15 >= abs(entroids[0,0]- x1) or 15 >= abs(entroids[0,2]- x1):
                            line1 = True
                        if 15 >= abs(entroids[1,0]- x1) or 15 >= abs(entroids[1,2]- x1):
                            line2 = True
                        if 15 >= abs(entroids[1,0]- entroids[0,0]) or 15 >= abs(entroids[1,2]- entroids[0,2]):
                            line2 = False
                    if line1 and line2:
                        check = True
                        #print(check)
                    if check:
                        sum1 = 0
                        for line in entroids:
                            x1, y1, x2, y2, = line
                            cv2.line(img, (x1, y1), (x2,y2), (0,255,0),3)
                        sum1 = sum1 + abs(entroids[0,0] - entroids[1,0])
                        sum1 = sum1 + abs(entroids[0,2] - entroids[1,2])
                        sum1 = sum1/2
                        
                        
                        pixGap = sum1
        apa = [90]
        if(wWidth[0] <= aWidth[0]):
            wWidth[0] = aWidth[0] + 1 
        if cvui.button(gui, 10, 410, 100,30,"Calibrate"):
            disPerPix = calibrate(wWidth[0],aWidth[0],pixGap)
        cvui.counter(gui, 10, 460, apa)   
       
        try: val = pixGap * disPerPix
        except:
            val = 0
        
        
        cv2.putText(img,str(pixGap),(10,30), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img,str("%.2f" %val),(100,30), font, 1,(255,255,255),2,cv2.LINE_AA)
      
        cv2.imshow("Camera", img)
        cv2.imshow("Canny", edges)
        cvui.update()
        cv2.imshow(WINDOW_NAME, gui)
         
        
if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
    
