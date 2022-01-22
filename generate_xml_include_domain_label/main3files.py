import os
import random

"""origin"""
#xmlfilepath=r'/home/dennischang/LRP-HAM/data/throat_uvula_dataset2007/VOC2007/Annotations'
#saveBasePath=r"/home/dennischang/LRP-HAM/data/throat_uvula_dataset2007/VOC2007/ImageSets/Main"

"""alex"""
xmlfilepath=r'/media/xuus/A45ED35B5ED324B8/LRP-HAM_git/generate_xml_include_domain_label/20211229_add_domain_label/target/xml'
saveBasePath=r"/media/xuus/A45ED35B5ED324B8/LRP-HAM_git/generate_xml_include_domain_label/20211229_add_domain_label/target/Main"

if not os.path.exists(saveBasePath):
    os.mkdir(saveBasePath)

trainval_percent=0.8
train_percent=0.7
total_xml = os.listdir(xmlfilepath)
# total_xml.sort(key= lambda x:int(x[1:-5]))
#print(total_xml)
num=len(total_xml)  
list=range(num)  
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)  
 
print("train and val size",tv)
print("train size",tr)
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,'train.txt'), 'w')  
fval = open(os.path.join(saveBasePath,'val.txt'), 'w')  
 
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

