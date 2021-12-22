#截一個資料夾內的影片變照片到另一個資料夾貼上
import cv2
import os
yourPath = './1090807上傳/'
outputPath = "./0807喉鏡/"
allFileList = os.listdir(yourPath)
for file in allFileList:
   print(file)

   #輸入影片，截圖，輸出照片
   vc = cv2.VideoCapture(yourPath+file) #读入视频文件
   os.makedirs(outputPath+file)
   c=1
 
   if vc.isOpened(): #判断是否正常打开
       rval , frame = vc.read()
   else:
       rval = False
 
   timeF = 1  #视频帧计数间隔频率
   try: 
       while rval:   #循环读取视频帧
           rval, frame = vc.read()
           if(c%timeF == 0): #每隔timeF帧进行存储操作
               cv2.imwrite(outputPath+file+'/'+str(c) + '.jpg',frame) #存储为图像
           c = c + 1
           cv2.waitKey(1)
       vc.release()
   except:
       print("end of",file)

