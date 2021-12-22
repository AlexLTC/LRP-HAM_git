#將combine裡所有的照片合成為一個影片
import cv2
import glob, os
movie_name = ['绒毛状腺瘤低级别上皮内瘤变横结肠2']
movie_number = [40]

for i in range(1):
    img_path = glob.glob("/home/dennischang/LRP-HAI/demo_result/*.jpg")
    #img_path = glob.glob("/home/xuus/LRP-HAI/demo_result/"+movie_name[i]+"/*.jpg")

    #img_path.sort()
    #print(img_path)
    img_path = sorted(img_path,key= lambda x:int(x[movie_number[i]:-4]))
    #print(img_path)
    video=cv2.VideoWriter('/home/dennischang/LRP-HAI/demo_result/video/'+movie_name[i]+'.mp4', cv2.VideoWriter_fourcc(*'MJPG'), 10, (497,374))

    for path in img_path:
         print(path)
         img  = cv2.imread(path)
         #cv2.imshow('img',img)
         #cv2.waitKey(100) 
         #img = cv2.resize(img,(128,128))
         video.write(img)

