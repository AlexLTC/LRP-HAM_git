import os 
import sys
# os.system('python ./print.py')

#def if_name():
#    if args.imdb_name == 'voc_2007_trainval':
#        path = './data/VOCdevkit2007/VOC2007/ImageSets'
#    else:
#       if args.imdb_name == 'polyp_2007_trainval':
#            path = './data/polyp_dataset2007/VOC2007/ImageSets'
#       elif args.imdb_name == 'throat_2007_trainval':
#            path = './data/throat_dataset2007/VOC2007/ImageSets'
#       else:
#            raise NotImplementedError
#    os.rename(os.path.join(path, 'Main' + str(args.num_kfolder)), os.path.join(path, 'Main'))


path = 'data/VOCdevkit2007/VOC2007/ImageSets'
os.rename(os.path.join(path, 'Main' + str(0)), os.path.join(path, 'Main'))
os.system('./experiments/scripts/train_P4.sh 0 pascal_voc res101 0')
os.rename(os.path.join(path, 'Main'), os.path.join(path, 'Main' + str(0)))

# os.rename(os.path.join(path, 'Main' + str(1)), os.path.join(path, 'Main'))
# os.system('./experiments/scripts/train_P4.sh 0 polyp res101 1')
# os.rename(os.path.join(path, 'Main'), os.path.join(path, 'Main' + str(1)))

#os.rename(os.path.join(path, 'Main' + str(2)), os.path.join(path, 'Main'))
#os.system('./experiments/scripts/train_P4.sh 0 polyp res101 2')
#os.rename(os.path.join(path, 'Main'), os.path.join(path, 'Main' + str(2)))

# os.rename(os.path.join(path, 'Main' + str(3)), os.path.join(path, 'Main'))
# os.system('./experiments/scripts/train_P4.sh 0 polyp res101 3')
# os.rename(os.path.join(path, 'Main'), os.path.join(path, 'Main' + str(3)))

# os.rename(os.path.join(path, 'Main' + str(4)), os.path.join(path, 'Main'))
# os.system('./experiments/scripts/train_P4.sh 0 polyp res101 4')
# os.rename(os.path.join(path, 'Main'), os.path.join(path, 'Main' + str(4)))

#os.system('~/xuus/LRP-HAI_p4/experiments/scripts/train_P4.sh 0 polyp res101 0')
#os.system('~/xuus/LRP-HAI_p4/experiments/scripts/train_P4.sh 0 polyp res101 1')
#os.system('~/xuus/LRP-HAI_p4/experiments/scripts/train_P4.sh 0 polyp res101 2')
#os.system('~/xuus/LRP-HAI_p4/experiments/scripts/train_P4.sh 0 polyp res101 3')
#os.system('~/xuus/LRP-HAI_p4/experiments/scripts/train_P4.sh 0 polyp res101 4')



