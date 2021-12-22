##深度学习过程中，需要制作训练集和验证集、测试集。
import main_value
import os, random, shutil
import re
print(main_value.real_file_dir)
def moveFileTo(fileDir ,tarDir):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        for name in pathDir:
                shutil.move(fileDir+name, tarDir+name)
        return

if __name__ == '__main__':
	#源图片文件夹路径
	fileDir_ori_6 = '/home/dennischang/Desktop/laryngonscope/測試結果/標記結果/聲帶/' 
  
	#源图片文件夹路径
	fileDir_tru_6 = '/home/dennischang/Desktop/laryngonscope/測試結果/標記結果/聲帶/'   

	tarDir_train_ori = '/home/dennischang/Desktop/1090422/output0521/'+str(main_value.real_file_dir)+'/'    #移动到新的文件夹路径
	tarDir_train_tru = '/home/dennischang/Desktop/1090422/output0521/'+str(main_value.final_location)+'/result_line/'    
	if not os.path.exists(tarDir_train_ori):
 	   os.mkdir(tarDir_train_ori)

	moveFileTo(fileDir_ori_6, tarDir_train_ori)

	#moveFileTo(fileDir_tru_6, tarDir_train_tru)

