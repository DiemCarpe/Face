import cv2 as cv
#读取图片
img=cv.imread('20190529100938.jpg')
#显示图片
cv.imshow('read_img',img)
#将图片灰度
gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray_img',gray_img)
#保存
cv.imwrite('gray_img.jpg',gray_img)
#等待键盘输入 单位毫秒 0是无线等待
cv.waitKey(0)
#底层是Opencv底层是C++所以需要释放内存
cv.destroyAllWindows()

import cv2 as cv
img = cv.imread('20190529100938.jpg')
cv.imshow('img', img)
print('原来的高度{}'.format(img.shape))
resize_img = cv.resize(img, dsize=(400, 300))
cv.imshow('resize_img.jpg', resize_img)
# 按q就关闭
while True:
    if ord('q') == cv.waitKey(0):
        break

cv.destroyAllWindows()




'''
绘制矩形,圆形
'''

import cv2 as cv

img=cv.imread('20190529100938.jpg')
#左上角坐标是x，y 矩形的宽度和高度是（w,h）
# x,y,w,h=100,100,100,100
#x,y,w,h color=BGR 矩形
# rectangle_img=cv.rectangle(img,(x,y,x+w,y+h),color=(0,255,0),thickness=2)

#center圆点的坐标 redius 半径
x,y,r=100,100,100
rectangl_img=cv.circle(img,center=(x,y),radius=r,color=(0,0,255),thickness=1)
cv.imshow('rectangle.jpg',rectangl_img)

cv.waitKey(0)
cv.destroyAllWindows()




'''

Haar获取人脸特征检测
'''

import cv2 as cv

def face_datect_demo():
    #首先将图片灰度
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #加载特征数据匹配什么特征
    face_detector=cv.CascadeClassifier('D:/Python/OpenCV/opencv/sources/data/haarcascades\\haarcascade_frontalface_default.xml')
    #可以获取图片的坐标
    faces=face_detector.detectMultiScale(gray)
    for x,y,w,h in faces:
        #绘制矩形
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=1)
    cv.imshow('result',img)

#加载图片
img=cv.imread('1.png')
face_datect_demo()
# cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()











'''
检测多张人脸
'''
import cv2 as cv
def face_detect_demo():
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #加载特征数据
    face_detector = cv.CascadeClassifier(
        'D:/Python/OpenCV/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    #detectMultiScale 调整参数scaleFactor缩放比例  minNeighbors至少检测多少次 maxSize检测区域最大的大小  minSize检测区域最小的大小
    faces=face_detector.detectMultiScale(gray,scaleFactor=1.01,minNeighbors=5,maxSize=(30,30))
    for x,y,w,h in faces:
        print(x,y,w,h)
        #绘制矩形
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=1)
        #绘制圆形
        cv.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(255,0,0),thickness=1)
    cv.imshow('ree',img)
img=cv.imread('5.png')
#调用函数
face_detect_demo()
while True:
    if ord('q') == cv.waitKey(0):
        break

cv.destroyAllWindows()







'''
视频中的人脸检测
'''
import cv2 as cv

#读取视频
def face_datect_demo(img):
    #将图片灰度
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #加载特征数据
    face_detector=cv.CascadeClassifier(
        'D:/Python/OpenCV/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml'
    )
    #
    faces=face_detector.detectMultiScale(gray)

    for x,y,w,h in faces:
        print(x,y,w,h)
        cv.rectangle(img,(x,y),(x+w,y+h),color=(255,0,0),thickness=1)
        cv.circle(img,center=(x+w//2,y+h//2),radius=(w//2),color=(0,255,0),thickness=2)
    cv.imshow('result',img)
#读取视频
video_cap=cv.VideoCapture('1.mp4')

while True:
    flag,frame=video_cap.read()
    print(flag,frame.shape)
    if not flag:
        break
    face_datect_demo(frame)
    if ord('q') ==cv.waitKey(1):
        break
#是否内存
cv.destroyAllWindows()
#释放视频的空间
video_cap.release()






'''
训练人脸识别
pip install opencv-contrib-python
'''

import cv2
import os
import sys
from  PIL import Image
import numpy as np

def getImageAndLabels(path):
    facesSamples=[]
    ids=[]
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #检测人脸
    # 加载特征数据
    face_detector = cv2.CascadeClassifier(
        'D:/Python/OpenCV/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml'
    )

    #遍历列表中的图片
    for imagePath in imagePaths:
        #打开图片
        PIL_img=Image.open(imagePath).convert('L')
        #将图像转换为数组
        img_np=np.array(PIL_img,'uint8')
        faces = face_detector.detectMultiScale(img_np)
        #获取每张图片的id 对路径进行切分

        id=int(os.path.split(imagePath)[1].split('.')[0])
        for x,y,w,h in faces:
            # print(x,y,w,h)
            #对图像进行切割 并放入
            facesSamples.append(img_np[y:y+h,x:x+w])
            ids.append(id)

            # print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
            # print(facesSamples,ids)
            # facess,idss=facesSamples,ids
            # print(faces,idss)
            return facesSamples,id
if __name__ == '__main__':
    #图片路径
    path='./data/jm/'
    #获取图像数组和id标签数组
    faces,ids=getImageAndLabels(path)
    # print('getImageAndLabels=================================================')
    # print(faces,ids)
    #获取训练对象
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(ids))
    #保存文件
    recognizer.write('trainer/trainer.yml')










