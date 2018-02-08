import cv2
import dlib
import numpy
import sys
import os
PREDICTOR_PATH = "F:\pywork\BareFootResearch-master\\testdeconv\shape_predictor_68_face_landmarks.dat"
savepath1="F:\pywork\database\\face\guanjiandian\celebA\color"
savepath2="F:\pywork\database\\face\guanjiandian\celebA\single"
# 1.使用dlib自带的frontal_face_detector作为我们的人脸提取器
detector = dlib.get_frontal_face_detector()

# 2.使用官方提供的模型构建特征提取器
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class NoFaces(Exception):
    pass
g = os.walk("F:\pywork\BareFootResearch-master\DCGAN\data\celebA\\temp")
k=0
for path,d,filelist in g:
    for filename in filelist:
        if filename.endswith('jpg'):
            if k<=5000:
                im = cv2.imread(os.path.join(path, filename))
                k=k+1
                sp = im.shape
# 3.使用detector进行人脸检测 rects为返回的结果
                rects = detector(im, 1)

# 4.输出人脸数，dets的元素个数即为脸的个数
                if len(rects) >= 1:
                    print("{} faces detected".format(len(rects)))

                    for i in range(len(rects)):
# 5.使用predictor进行人脸关键点识别
                        landmarks = numpy.matrix([[p.x, p.y] for p in predictor(im, rects[i]).parts()])
                        im = im.copy()
                        image = numpy.zeros(im.shape, numpy.uint8)
# 使用enumerate 函数遍历序列中的元素以及它们的下标
                        for idx, point in enumerate(landmarks):
                            pos = (point[0, 0], point[0, 1])
                            if 0<(point[0, 0] + 2)< sp[1] and 0<(point[0, 1] + 2) < sp[0]:
                                if 0<(point[0, 0] - 2)< sp[1] and 0<(point[0, 1] - 2)<sp[0]:
                                    for i in range(2):
                                        image[point[0, 1] - i, point[0, 0] ] = 255
                                        image[point[0, 1] + i, point[0, 0] ] = 255
                                        image[point[0, 1] , point[0, 0] - i] = 255
                                        image[point[0, 1] , point[0, 0] + i] = 255
# 6.绘制特征点
                            cv2.circle(im, pos, 3, color=(0, 255, 0))
                    cv2.imwrite(os.path.join(savepath1, filename),im)
                    cv2.imwrite(os.path.join(savepath2, filename), image)

                # cv2.namedWindow("im", 0)
                # cv2.resizeWindow("im", sp[1], sp[0]);
                # cv2.imshow("im", im)
                # cv2.waitKey(0)
                # cv2.namedWindow("im1", 0)
                # cv2.resizeWindow("im1", sp[1], sp[0]);
                # cv2.imshow("im1", image)
                # cv2.waitKey(0)

