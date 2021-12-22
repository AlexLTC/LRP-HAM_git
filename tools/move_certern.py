import os, random, shutil
import re
def moveFile(fileDir):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        filenumber=len(pathDir)
        for i in range(2001,3001):#range(11865,11868)=>11865~11867
             name = str(i)+'.jpg'
             shutil.move(fileDir+name, tarDir+name)
        return

if __name__ == '__main__':
	fileDir = "/home/dennischang/Desktop/1090422/movie_pic/20200423_085417/"    #源图片文件夹路径
	tarDir = "/home/dennischang/Desktop/1090422/movie_pic/20200423_085417_2/"    #移动到新的文件夹路径

	moveFile(fileDir)
