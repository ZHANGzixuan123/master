#直线检测
#使用霍夫直线变换做直线检测，前提条件：边缘检测已经完成
import cv2 as cv
import numpy as np

#标准霍夫线变换
def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  #apertureSize参数默认其实就是3
    cv.imshow("edges", edges)
    lines = cv.HoughLines(edges, 1, np.pi/180, 80)
    for line in lines:
        rho, theta = line[0]  #line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)   #theta是弧度
        b = np.sin(theta)
        x0 = a * rho    #代表x = r * cos（theta）
        y0 = b * rho    #代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b)) #计算直线起点横坐标
        y1 = int(y0 + 1000 * a)    #计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b)) #计算直线终点横坐标
        y2 = int(y0 - 1000 * a)    #计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)    #点的坐标必须是元组，不能是列表。
    cv.imshow("image-lines", image)

#统计概率霍夫线变换
def line_detect_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray = cv.medianBlur(gray,5)
    edges = cv.Canny(gray, 50, 255, apertureSize=3)  # apertureSize参数默认其实就是3
    lines = cv.HoughLinesP(edges, 1.0, np.pi / 180, 220,1, minLineLength=20, maxLineGap=9)
    print (len(lines))
    #lines1 = lines[:,0,:]
    #for x1,y1,x2,y2 in lines1[:]: 
    for line in lines:
        x1, y1, x2, y2 = line[0]
     
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        ab = (abs(x1-x2),abs(y1-y2))
        cd = (abs(x3-x4),abs(y3-y4))
        xDis = x2 - x1  #x的增量
        yDis = y2 - y1  #y的增量
        if(abs(xDis) > abs(yDis)):
            maxstep = abs(xDis)
        else:
            maxstep = abs(yDis)
        xUnitstep = xDis/maxstep  #x每步骤增量
        yUnitstep = yDis/maxstep  #y的每步增量
        x = x1
        y = y1
        for k in range(maxstep):
            x = x + xUnitstep
            y = y + yUnitstep
           # print("x: %d, y:%d" % (x, y))

    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imshow("line_detect_possible_demo",image)
    
    

src = cv.imread('U:\computer.jpg')
src = cv.resize(src,(450,450),interpolation = cv.INTER_CUBIC)
print(src.shape)
cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE) 
cv.imshow('input_image', src)
line_detection(src)
src = cv.imread('U:\computer.jpg') #调用上一个函数后，会把传入的src数组改变，所以调用下一个函数时，要重新读取图片
src = cv.resize(src,(450,450),interpolation = cv.INTER_CUBIC)
line_detect_possible_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
