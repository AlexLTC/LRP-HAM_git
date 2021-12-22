# 選擇不同的照片
import cv2
import numpy as np
import os
from PIL import Image
from PIL import ImageChops
import operator
from itertools import chain


def pHash(img_name):
 """
 get image pHash value
 """
 # 加载并调整图片为32x32灰度图片
 img = cv2.imread(img_name, 0)
 img = cv2.resize(img, (640, 320), interpolation=cv2.INTER_CUBIC)

 # 创建二维列表
 h, w = img.shape[:2]
 vis0 = np.zeros((h, w), np.float32)
 vis0[:h, :w] = img # 填充数据

 # 二维Dct变换
 vis1 = cv2.dct(cv2.dct(vis0))
 vis1.resize((640, 320), refcheck=False)

 # 把二维list变成一维list
 img_list = list(chain.from_iterable(vis1))

 # 计算均值
 avg = sum(img_list) * 1. / len(img_list)
 avg_list = ['0' if i < avg else '1' for i in img_list]

 # 得到哈希值
 return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 640 * 320, 4)])


def hammingDist(s1, s2):
 """
 计算两张图片的汉明距离
 """
 assert len(s1) == len(s2)
 return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


picture_num = len(os.listdir('/home/dennischang/LRP-HAI/tools/video/Support Vector Machines, Clearly Explained!!!.mp4/'))
for i in range(6197,picture_num+6198):
   #print(i)
   num=str(i) 
   num1=str(i+1) 
   #img1=cv2.imread('/home/dennischang/LRP-HAI/tools/video/Support Vector Machines, Clearly Explained!!!.mp4/'+num+'.jpg',cv2.IMREAD_COLOR)
   img2=cv2.imread('/home/dennischang/LRP-HAI/tools/video/Support Vector Machines, Clearly Explained!!!.mp4/'+num1+'.jpg',cv2.IMREAD_COLOR)
   #difference = cv2.subtract(img1, img2)
   #result = not np.any(difference) #if difference is all zeros it will return False

   #t1=Image.open('/home/dennischang/LRP-HAI/tools/video/Support Vector Machines, Clearly Explained!!!.mp4/'+num+'.jpg')
   #t2=Image.open('/home/dennischang/LRP-HAI/tools/video/Support Vector Machines, Clearly Explained!!!.mp4/'+num1+'.jpg')
   #result=operator.eq(t1,t2)
   HASH1 = pHash('/home/dennischang/LRP-HAI/tools/video/Support Vector Machines, Clearly Explained!!!.mp4/'+num+'.jpg')
   HASH2 = pHash('/home/dennischang/LRP-HAI/tools/video/Support Vector Machines, Clearly Explained!!!.mp4/'+num1+'.jpg')
   distance = hammingDist(HASH1, HASH2)
   #print('汉明距离=%f' % distance)
   out_score = 1 - distance * 1. / (640 * 320 / 4)
   print('pic_num'+num+'相似度=%f' % out_score)

   if out_score < 0.99:
      cv2.imwrite('/home/dennischang/下載/Support Vector Machines, Clearly Explained!!!/h/'+num1+'.jpg',img2)
      print(out_score)
print('finished')

