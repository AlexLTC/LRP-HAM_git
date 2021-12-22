#將四種照片拼接成一個照片
import cv2
import numpy as np
import os
pic_file = '20200421_食道and氣道and模糊）'
picture_num = len(os.listdir('/home/dennischang/Desktop/1090422/output0503/'+pic_file+'/result_line/'))
for i in range(1,picture_num+1):#166
   print(i)
   num=str(i) 
   img1=cv2.imread('/home/dennischang/Desktop/1090422/output0503/'+pic_file+'/result_line/'+num+'.jpg',cv2.IMREAD_COLOR)
   img2=cv2.imread('/home/dennischang/Desktop/1090422/output0503/output_data/'+pic_file+'/result_line/'+num+'.jpg',cv2.IMREAD_COLOR)

   if img1 is None:
      print('continue')
      i=i+1
      continue

   img=np.concatenate((img1,img2),axis=1)    #x轴方向拼接
   #concated_img2=np.concatenate((img3,img4),axis=1)   #x轴方向拼接
   #img=np.concatenate((concated_img,concated_img2),axis=0)   #y轴方向

   #img=np.concatenate((img1,img2),axis=0)
   cv2.imwrite('/home/dennischang/Desktop/1090422/movie_pic_threshold/'+pic_file+'/'+num+'.jpg',img)

   print('/home/dennischang/Desktop/1090422/movie_pic_threshold/'+pic_file+'/'+num+'.jpg saved!')

print('finished')
