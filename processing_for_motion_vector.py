import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv
from scipy import ndimage
from math import sqrt
import os


flag = 1

def setPixel(data,X,Y,accumulate_flag):
    xb = int((data[6]-data[2]/2)/8)
    yb = int((data[7]-data[3]/2)/8)
    xw = int(data[2]/8)
    yw = int(data[3]/8)
    #print(xb,yb,xw,yw)
    #print(data[8],data[9])
    for i in range(int(xw)):
        for j in range(int(yw)):
            #print(xb+i,yb+j)
            if accumulate_flag =='GOP' or 'True':
                X[yb+j,xb+i] += abs(data[8])
                Y[yb+j,xb+i] += abs(data[9])
            else:
                X[yb+j,xb+i] = abs(data[8])
                Y[yb+j,xb+i] = abs(data[9])

def modulus(X, Y):
    mod = np.sqrt(np.power(X, 2) + np.power(Y, 2))
    # mod[mod<=threshold] = 0
    # mod[mod>threshold] = 255
    return mod

def angle(X,Y,X_avg,Y_avg):
    ang = np.zeros([64,128])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            ## cos(theta)
            a = (X[i,j]*X_avg[i,j]+Y[i,j]*Y_avg[i,j])/sqrt((X[i,j]*X[i,j]+Y[i,j]*Y[i,j])*(X_avg[i,j]*X_avg[i,j]+Y_avg[i,j]*Y_avg[i,j]))
            if a >= -1 and a <= 1:
                ang[i, j] = a

    return ang

def main():
    for i in [3, 5, 6, 7, 8, 9, 10]:
        fileName = 's1p2_'+str(i)
        if not os.path.exists('./mask/'+fileName):
            os.mkdir('./mask/'+fileName)
        data = pd.read_csv('../mv/%s.csv'%fileName)
        data = np.array(data)
        video = []
        ####### configuration section
        height = 64
        width = 128
        GOP_size = 5
        beginIndex = data[0,0]
        endIndex = data[-1,0]
        i = 0
        ####### whether or not accumulate the motion vector
        accumulate_flag = 'GOP'
        ####### accumulate X,Y motion vector
        '''
        if accumulate_flag == 'True' and accumulate_flag != 'GOP':
            X = np.zeros([height,width])
            Y = np.zeros([height,width])
        '''
        print('visualizing begin')
        for index in range(beginIndex,endIndex):
            ####### calculate X,Y motion vector each time
            print(index)
            if accumulate_flag == 'GOP' and ((index-1) % GOP_size == 0 or index == beginIndex):
                X = np.zeros([height,width])
                Y = np.zeros([height,width])
            while(data[i,0] == index):
                setPixel(data[i,:],X,Y,accumulate_flag=accumulate_flag)
                i+=1
            video.append([X,Y])
            print(index)
            if index % GOP_size == 0:

                kernel_3x3 = np.array([
                    [1 / 8, 1 / 8, 1 / 8],
                    [1 / 8, 0, 1 / 8],
                    [1 / 8, 1 / 8, 1 / 8]])
                X_avg = ndimage.convolve(X, kernel_3x3)
                Y_avg = ndimage.convolve(Y, kernel_3x3)

                ang = angle(X, Y, X_avg, Y_avg)
                mod = modulus(X, Y)
                for m in range(ang.shape[0]):
                    for n in range(ang.shape[1]):
                        if 0.5 * ang[m, n] + 0.5 * mod[m, n] > 1:
                            ang[m, n] = 255
                        else:
                            ang[m, n] = 0
                # kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
                # kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
                # kernel3 = cv.getStructuringElement(cv.MORPH_RECT, (13, 13))
                #
                # ang = cv.morphologyEx(ang, cv.MORPH_OPEN, kernel1)
                # ang = cv.dilate(ang, kernel1)
                # ang = cv.morphologyEx(ang, cv.MORPH_CLOSE, kernel1)
                #ang = cv.morphologyEx(ang, cv.MORPH_CLOSE, kernel2)
                #ang = cv.morphologyEx(ang, cv.MORPH_CLOSE, kernel3)


                #ang = ang * 255
                cv.imwrite('./mask/%s/%s_%i.png' % (fileName, fileName, index-1), ang)  # 保存图片
                cv.imwrite('./mask/%s/%s_%i.png' % (fileName, fileName, index-2), ang)  # 保存图片
                cv.imwrite('./mask/%s/%s_%i.png' % (fileName, fileName, index-3), ang)  # 保存图片
                cv.imwrite('./mask/%s/%s_%i.png' % (fileName, fileName, index-4), ang)  # 保存图片
                cv.imwrite('./mask/%s/%s_%i.png' % (fileName, fileName, index-5), ang)  # 保存图片


if __name__ == '__main__':
    main()





