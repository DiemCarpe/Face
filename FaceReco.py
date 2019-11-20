'''

人脸识别

'''
import cv2
import numpy as np
import os
import datetime
from PIL import Image


class FaceRecognition():
    # 训练人脸识别 产生文件提供给图像处理使用 需要传入图片路径
    def getImageAndLabels(self, path):
        # 存放切割的图像
        facesSamples = []
        ids = []
        # 获取训练人脸识别图片的路径
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        # 加载特征数据
        face_detector = cv2.CascadeClassifier(
            'D:/Python/OpenCV/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml'
        )
        # 遍历列表中的图片
        for imagePath in imagePaths:
            # 利用PIL的Image.open方法打开图片
            PIL_img = Image.open(imagePath).convert('L')
            # 将图片转换为数组
            img_np = np.array(PIL_img, 'uint8')
            faces = face_detector.detectMultiScale(img_np)
            # 获取每张图片的id 对路径进行切分
            id = int(os.path.split(imagePath)[1].split('.')[0])
            for x, y, w, h in faces:
                # print(x,y,w,h)
                # 对图像进行切割并放入列表中
                facesSamples.append(img_np[y:y + h, x:x + w])
                # 将图片ID存放进列表中
                ids.append(id)
                # 获取训练对象
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(ids))
            # 保存文件
            recognizer.write('trainer/trainer.yml')

    # 处理图像并判断置信评分
    def face_datect_demo(self, img):
        # 加载并读取训练数据集文件
        recogizer = cv2.face.LBPHFaceRecognizer_create()
        recogizer.read('trainer/trainer.yml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier(
            'D:/Python/OpenCV/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml'
        )
        faces = face_detector.detectMultiScale(gray)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.circle(img, center=(x + w // 2, y + h // 2), radius=(w // 2), color=(0, 0, 255), thickness=1)
            # 进行人脸识别
            id, confidences = recogizer.predict(gray[y:y + h, x:x + w])
            confidence = int(str(confidences).split('.')[0])
            # 返回置信分与id
            # return id,confidence
            # 返回置信分
            return confidence

    # 调用摄像头拍照并保存图片
    def Camera(self):
        # 调用笔记本的主摄像头
        cap = cv2.VideoCapture(0)
        # 拍摄图片
        ret, frame = cap.read()
        now = str(datetime.datetime.now())[12:19].replace(':', '')
        # 根据时间命名图片
        fn = 'data/photo/' + now[:10] + now + '.jpg'
        cv2.waitKey(1)
        # 保存图片
        cv2.imwrite(fn, frame)
        # 释放相机资源
        cap.release()
        return fn


if __name__ == '__main__':
    fr = FaceRecognition()
    # 加载拍摄的图片
    # img = cv2.imread(fr.img_ren())
    fr.face_datect_demo(cv2.imread(fr.Camera()))
    # cv2.waitKey(0)
    # 释放内存
    cv2.destroyAllWindows()
