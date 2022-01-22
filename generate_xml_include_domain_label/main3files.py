import os
import random

"""origin"""
#xmlfilepath=r'/home/dennischang/LRP-HAM/data/throat_uvula_dataset2007/VOC2007/Annotations'
#saveBasePath=r"/home/dennischang/LRP-HAM/data/throat_uvula_dataset2007/VOC2007/ImageSets/Main"

"""alex"""
xmlfilepath=r'/media/xuus/A45ED35B5ED324B8/alex/for_train_data/20220115/target/xml'
saveBasePath=r"/media/xuus/A45ED35B5ED324B8/alex/for_train_data/20220115/Main"

trainval_percent=0.8
train_percent=0.7
total_xml = os.listdir(xmlfilepath)
# total_xml.sort(key= lambda x:int(x[1:-5]))
#print(total_xml)
num=len(total_xml)  
list=range(num)

if "source_" in total_xml[0]:
    tv = num
else:  # target
    tv=int(num*trainval_percent)

tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
 
print("train and val size",tv)
print("train size",tr)

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

