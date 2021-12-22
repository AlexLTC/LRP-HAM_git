import os
import main_value
plus_num = int(main_value.plus_num_main)

def batch_rename(path):
    for fname in os.listdir(path):
        new_fname = fname.split(".")[0]
        print('new_fname',new_fname)
        new_fname = int(new_fname)+plus_num

        #new_fname = str(new_fname)+'.'+fname.split(".")[1]+'.'+fname.split(".")[2]
        new_fname = str(new_fname)+'.jpg'#+fname.split(".")[1]+'.'+fname.split(".")[2]
        print('new_fname',new_fname,'ORIGINAL',os.path.join(path, fname))
        os.rename(os.path.join(path, fname), os.path.join(path, new_fname))

path = '/home/dennischang/Desktop/laryngonscope/測試結果/標記結果/聲帶'
batch_rename(path)
#path = '/home/dennischang/Desktop/02_25/code/gcn/result'
#batch_rename(path)
