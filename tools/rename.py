import os, random, shutil
# import main_value
# plus_num = int(main_value.plus_num_main)


def batch_plus_rename(path):
    for fname in os.listdir(path):
        new_fname = fname.split(".")[0]
        new_fname = int(new_fname)+plus_num
        new_fname = str(new_fname)+'.jpg'#+fname.split(".")[1]+'.'+fname.split(".")[2]
        os.rename(os.path.join(path, fname), os.path.join(path, new_fname))

def moveFileTo(fileDir, tarDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    for name in pathDir:
        shutil.move(fileDir + name, tarDir + name)
    return


if __name__ == '__main__':
    while False:
        plus_num = 1500
        for i in range(1,70):
            path = '/home/dennischang/1090807_Laryngoscopy/('+str(i)+')'
            path_all = '/home/dennischang/1090807_Laryngoscopy/all'

            for pic_name in os.listdir(path+'/ori/'):
                plus_num = plus_num + 1
                new_fname = '('+str(plus_num)+').jpg'
                pic_name_nbr = pic_name.split(')')[0].split('(')[1]
                shutil.copy(path+'/ori/' + pic_name, path_all+'/ori/' + pic_name_nbr+'.jpg')
                os.rename(os.path.join(path_all+'/ori/', pic_name_nbr +'.jpg'), os.path.join(path_all+'/ori/', new_fname))

                shutil.copy(path+'/會厭/' + pic_name, path_all+'/會厭/' + pic_name_nbr+'.jpg')
                os.rename(os.path.join(path_all+'/會厭/', pic_name_nbr+'.jpg'), os.path.join(path_all+'/會厭/', new_fname))

                shutil.copy(path+'/杓狀/' + pic_name, path_all+'/杓狀/' + pic_name_nbr+'.jpg')
                os.rename(os.path.join(path_all+'/杓狀/', pic_name_nbr+'.jpg'), os.path.join(path_all+'/杓狀/', new_fname))

                shutil.copy(path+'/聲帶/' + pic_name, path_all+'/聲帶/' + pic_name_nbr+'.jpg')
                os.rename(os.path.join(path_all+'/聲帶/', pic_name_nbr+'.jpg'), os.path.join(path_all+'/聲帶/', new_fname))
    while True:
        path = '/media/data/VOCdevkit/三合一/argumentation2'
        tar_path = '/home/dennischang/1090807_Laryngoscopy/all2'
        shutil.copy(path + '/ori', tar_path)

