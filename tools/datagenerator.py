from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import cv2
import os
from skimage import io, transform  # python的Image和skimage处理图片

### path
data_path = '/home/dennischang/1090807_Laryngoscopy/all2/ori/'
label_path1 = '/home/dennischang/1090807_Laryngoscopy/all2/會厭/'
label_path2 = '/home/dennischang/1090807_Laryngoscopy/all2/杓狀/'
label_path3 = '/home/dennischang/1090807_Laryngoscopy/all2/聲帶/'
### parameters
w = 640
h = 480
c = 3

### read the data 
def read_img(path):
    imgs = []
    for filename in os.listdir(path):
        print(filename)
        img = io.imread(path + filename)  
        x = transform.resize(img, (h, w, c))   # this is a Numpy array with shape (w, h, 3))
        imgs.append(x)    # this is a Numpy array with shape (images_number, w, h, 3)
    return np.asarray(imgs, np.float32)     

data = read_img(data_path)
label1 = read_img(label_path1)
label2 = read_img(label_path2)
label3 = read_img(label_path3)

### ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

seed = 1
i = 1
for batch in datagen.flow(data, batch_size=1,seed=seed,
                          save_to_dir='/home/dennischang/1090807_Laryngoscopy/all3/ori/', save_format='jpg'):
    i += 1
    print('1',i)
    if i > 9264:
        break  
        
i = 1
for batch in datagen.flow(label1, batch_size=1,seed=seed,
                          save_to_dir='/home/dennischang/1090807_Laryngoscopy/all3/會厭/', save_format='jpg'):
    i += 1
    print('2',i)
    if i > 9264:
        break
i = 1
for batch in datagen.flow(label2, batch_size=1,seed=seed,
                          save_to_dir='/home/dennischang/1090807_Laryngoscopy/all3/杓狀/', save_format='jpg'):
    i += 1
    print('3',i)

    if i > 9264:
        break
i = 1
for batch in datagen.flow(label3, batch_size=1,seed=seed,
                          save_to_dir='/home/dennischang/1090807_Laryngoscopy/all3/聲帶/', save_format='jpg'):
    i += 1
    print('4',i)
    if i > 9264:
        break
'''
import numpy as np


size = (2560, 1600)
# 全黑.可以用在屏保
black = np.zeros(size)
print(black[34][56])
cv2.imwrite('black.jpg',black)

#white 全白
black[:]=255
print(black[34][56])
cv2.imwrite('white.jpg',black)
'''
