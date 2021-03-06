import os, sys
import glob
from PIL import Image
"""origin"""
# path = '/home/dennischang/LRP-HAM/data/throat_uvula_dataset2007/VOC2007/20210116 Label/argument/'

"""alex"""
path = '/media/xuus/A45ED35B5ED324B8/KITTI_dataset/data_object_label_2/training/'

# VEDAI 图像存储位置
# src_img_dir = path + 'gt'
src_img_dir = '/media/xuus/A45ED35B5ED324B8/KITTI_dataset/data_object_image_2/training/image_2_jpg'
# VEDAI 图像的 ground truth 的 txt 文件存放位置
src_txt_dir = path + "txt"
src_xml_dir = path + "xml"  # output path

if not os.path.exists(src_xml_dir):
    os.mkdir(src_xml_dir)
 
img_Lists = glob.glob(src_img_dir + '/*.jpg')
 
img_basenames = [] # e.g. 100.jpg
for item in img_Lists:
    img_basenames.append(os.path.basename(item))
 
img_names = [] # e.g. 100
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)
 
for img in img_names:
    im = Image.open((src_img_dir + '/' + img + '.jpg'))
    width, height = im.size
 
    # open the crospronding txt file
    gt = open(src_txt_dir + '/' + img + '.txt').read().splitlines()
    #gt = open(src_txt_dir + '/gt_' + img + '.txt').read().splitlines()
 
    # write in xml file
    #os.mknod(src_xml_dir + '/' + img + '.xml')
    xml_file = open((src_xml_dir + '/' + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + 'source_' + str(img) + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')
 
    # write the region of image on xml file
    for img_each_label in gt:
        spt = img_each_label.split(',') #这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
        #for i in range (0,4):
        if int(spt[3]) <= int(spt[1])  :
        #   print(img)
        #else:
          print(img)
        xml_file.write('    <object>\n')
        xml_file.write('        <name>' + str(spt[4]) + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(spt[0]) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(spt[1]) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(spt[2]) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(spt[3]) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')
 
    xml_file.write('</annotation>')
'''————————————————
版权声明：本文为CSDN博主「xunan003」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/xunan003/article/details/79052288'''
