import os 
import sys
# os.system('python ./print.py')
path = 'data/throat_dataset2007/VOC2007/ImageSets'

for i in range(3,4):
    os.rename(os.path.join(path, 'Main' + str(i)), os.path.join(path, 'Main'))
    os.system('./experiments/scripts/train_P4.sh 0 throat res101 '+str(i))
    os.rename(os.path.join(path, 'Main'), os.path.join(path, 'Main' + str(i)))

# os.system('./experiments/scripts/train_P4.sh 0 throat res101 0')
# os.system('./experiments/scripts/train_P4.sh 0 throat res101 1')
# os.system('./experiments/scripts/train_P4.sh 0 throat res101 2')
# os.system('./experiments/scripts/train_P4.sh 0 throat res101 3')
# os.system('./experiments/scripts/train_P4.sh 0 throat res101 4')


