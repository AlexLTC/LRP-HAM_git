import cv2


#輸入影片，截圖，輸出照片
vc = cv2.VideoCapture('./test.mp4') #读入视频文件
c=1
 
if vc.isOpened(): #判断是否正常打开
    rval , frame = vc.read()
else:
    rval = False
 
timeF = 1  #视频帧计数间隔频率
 
while rval:   #循环读取视频帧
    rval, frame = vc.read()
    if(c%timeF == 0): #每隔timeF帧进行存储操作
        cv2.imwrite('/home/dennischang/LRP-HAI/tools/video/test/'+str(c)+'.jpg',frame) #存储为图像
        # print(c)
    c = c + 1
    cv2.waitKey(1)
vc.release()
