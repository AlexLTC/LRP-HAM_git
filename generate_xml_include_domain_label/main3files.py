import os
import random

"""origin"""
#xmlfilepath=r'/home/dennischang/LRP-HAM/data/throat_uvula_dataset2007/VOC2007/Annotations'
#saveBasePath=r"/home/dennischang/LRP-HAM/data/throat_uvula_dataset2007/VOC2007/ImageSets/Main"

"""alex"""
xmlfilepath=r'/media/xuus/A45ED35B5ED324B8/DATE-FRCNN/data/polyp_dataset2007/VOC2007/Annotations'
saveBasePath=r'/media/xuus/A45ED35B5ED324B8/DATE-FRCNN/data/polyp_dataset2007/VOC2007/ImageSets/Main/for_ori_DA'

if not os.path.exists(saveBasePath):
    os.makedirs(saveBasePath)

trainval_percent=0.8
train_percent=0.7
total_xml = os.listdir(xmlfilepath)
# total_xml.sort(key= lambda x:int(x[1:-5]))
#print(total_xml)

# source or target 的檔名都是由數字排序,因此擷取數字排序與分類即可
num=len(total_xml)
# assert num==293, "file numbers error"
list=range(num)

# # source 只有在 trainval 部份，而 target 才會分佈在 trainval 與 test 中
# if 'source_' in total_xml:
#     trainval_size = int(num*trainval_percent)
# else:  # target
#     trainval_size =int(num*trainval_percent)  # trainval 佔所有 xml data 數量的 0.8
trainval_size = int(num*trainval_percent)
train_size = int(trainval_size*train_percent)  # train 佔 trainval xml data 檔案數量 0.7
trainval = random.sample(list,trainval_size)  
train=random.sample(trainval,train_size)  
 
print("train and val size",trainval_size)
print("train size",train_size)

if not os.path.exists(saveBasePath):
    os.mkdir(saveBasePath)
    ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
else:
    ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'a')  
    ftest = open(os.path.join(saveBasePath,'test.txt'), 'a')  
    ftrain = open(os.path.join(saveBasePath,'train.txt'), 'a')  
    fval = open(os.path.join(saveBasePath,'val.txt'), 'a')  
 
for i in list:
    name=total_xml[i][:-4]+'\n'  
    if i in trainval:  
        ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        ftest.write(name)  
  
ftrainval.close()  
ftrain.close()  
fval.close()  
ftest .close()

