###  for ploting fr-rcnn weights & ham weights loss ###
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# hai_losses_857 = []
# hai_losses_1009 = []
# hai_losses_1139 = []
# hai_losses_3597 = []
# hai_losses_4783 = []
#
# det_losses_857 = []
# det_losses_1009 = []
# det_losses_1139 = []
# det_losses_3597 = []
# det_losses_4783 = []

# cls_losses = []
# box_losses = []
# det_losses = []
# mean_losses = []
# rcnn_totLoss = []
# rcnn_rpnLossCls =[]
# rcnn_rpnLossBox =[]

ori_fr = []
tf_drop_fr = []
tf_noDrop_fr = []

ori_hai = []
tf_drop_hai =[]
tf_noDrop_hai = []

# input filename
print("plot filename:")
input_filename = input()  # frcnn or hai

# output error when input incorrect filename
if not input_filename == 'fr-rcnn' and not input_filename == 'hai' and not input_filename == 'pascal_voc':
    sys.exit('Error: Unavailable filename')

# define path
ori_path = '/media/xuus/A45ED35B5ED324B8/LRP-HAM/transfer_output/pascal_voc'
filename_path = ['original']  # original, dropout, no_dropout
concated_paths = [os.path.join(ori_path, filename) for filename in filename_path]
ori_fr_path = 'fr-rcnn-weights/P4/res101/voc_2007_trainval/default/log_train.txt' 
fr_path = 'fr-rcnn-weights/P4/res101/voc_2007_trainval/default/log_train.txt'
hai_path = 'HAM-weights/pascal_voc/P4/res101/log_train.txt'

# concate path & filename
if input_filename == 'fr-rcnn':
    concated_paths = [os.path.join(concated_path, fr_path) for concated_path in concated_paths]
elif input_filename == 'hai':
    concated_paths = [os.path.join(concated_path, hai_path) for concated_path in concated_paths]
else:
    sys.exit('Error: Unavailable filepath')

# start finding loss
for path in concated_paths:
    if os.path.exists(path):
        print("current path:", path)
    else:
        sys.exit("Error: path does not exists")

    with open(path, 'r') as f:
        line = f.readline()
        while line:
            if input_filename == 'hai':
                if line.find('Mean loss') != -1:
                    losses = line[-23:]
                   
                    # find tot
                    tot_loss = float(losses[losses.find('(')+1:losses.find(',')])
                
                    # find ma
                    # ma_loss = losses[losses.find(',')+1:losses.find(')')]
                
                    # mean_loss = losses[losses.find(',')+2:losses.find(')')]
                    if 'original' in path:
                        print(path)
                        ori_hai.append(float(tot_loss))
                    elif 'no_dropout' in path:
                        print(path)
                        tf_noDrop_hai.append(float(tot_loss))
                    elif 'dropout' in path:
                        print(path)
                        tf_drop_hai.append(float(tot_loss))
                    else:
                        print('Error: wrong file name')
                        exit()
                    # mean_losses.append(float(mean_loss))

            elif input_filename == 'fr-rcnn':
                if line.find('total loss') != -1:
                    num_totalLoss = float(line[-8:])
                    if 'original' in path:
                        ori_fr.append(float(num_totalLoss))
                    elif 'no_dropout' in path:
                        tf_noDrop_fr.append(float(num_totalLoss))
                    elif 'dropout' in path:
                        tf_drop_fr.append(float(num_totalLoss))
                    else:
                        print('wrong file name')
                        exit()
            else:
                sys.exit('fail in loop')

            line = f.readline()

# print(len(det_losses_857))
# print(len(det_losses_1193))
# print(len(det_losses_3597))
# print(len(det_losses_4783))
# print(len(mean_losses))
# print(len(range(50, 110050, 50)))
# print(len(cls_losses))
# print(len(box_losses))
# print(len(range(40000, 110000, 50)))
# print(len(total_losses))

# print(len(rcnn_totLoss))
# print(len(range(20, 70020, 20)))

# # for fr-rcnn use
# value1 = range(20, 70020, 20)
# plt.figure()
# # plt.plot(value1, rcnn_totLoss, color='r', label='rcnn tot loss(4783 images)')
# plt.plot(value1, rcnn_rpnLossCls, color='r', label='rcnn rpn cls loss(4783 images)')
# # plt.plot(value1, rcnn_rpnLossBox, color='g', label='rcnn box cls loss(4783 images)')
# plt.xlabel('iter')
# plt.ylabel('loss')
# plt.text((rcnn_rpnLossCls.index(min(rcnn_rpnLossCls))+1)*20, min(rcnn_rpnLossCls), min(rcnn_rpnLossCls), ha='center', va='bottom')
# plt.text(70000, rcnn_rpnLossCls[-1], rcnn_rpnLossCls[-1], ha='center', va='bottom')
# plt.legend()
# plt.show()

# for ham weights use
value1 = range(50, 110050, 50)
value2 = range(40000, 110000, 50)
value3 = range(20, 70020, 20)
plt.figure()
# plt.plot(value2, cls_losses, color='g', label='cls_loss')
# plt.plot(value2, box_losses, color='y', label='box_loss')
if input_filename == 'fr-rcnn':
    plt.plot(value3, ori_fr, color='r', label='Original_FR')
    # plt.plot(value3, tf_noDrop_fr, color='b', label='Transferred_NoDropout_FR')
    # plt.plot(value3, tf_drop_fr, color='g', label='Transferred_Dropout_FR')
elif input_filename == 'hai':
    plt.plot(value1, ori_hai, color='r', label='Original ARM Total loss')
    # plt.plot(value1, tf_noDrop_hai, color='b', label='Transferred NoDropout ARM Toatl loss')
    # plt.plot(value1, tf_drop_hai,  color='g', label='Transferred Dropout ARM Total loss')
else:
    print('Error: fail in plot')

plt.xlabel('iter')
plt.ylabel('loss')
# plt.text((hai_losses.index(min(hai_losses))+1)*50, min(hai_losses), min(hai_losses), ha='center', va='bottom')
# plt.text(110000, det_losses_857[-1], det_losses_857[-1], ha='center', va='bottom', fontsize=15)
# plt.text(110000, det_losses_1193[-1], det_losses_1193[-1], ha='center', va='bottom', fontsize=15)
# plt.text(110000, det_losses_3597[-1], det_losses_3597[-1], ha='center', va='bottom', fontsize=15)
# plt.text(110000, det_losses_4783[-1], det_losses_4783[-1], ha='center', va='bottom', fontsize=15)
plt.legend(fontsize=15)
plt.show()
