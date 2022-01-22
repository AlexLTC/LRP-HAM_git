import os, glob, cv2

"""origin path"""
# path = '/home/dennischang/LRP-HAM/data/throat_uvula_dataset2007/VOC2007/20210116 Label/argument/gt/'
# path_txt = '/home/dennischang/LRP-HAM/data/throat_uvula_dataset2007/VOC2007/20210116 Label/argument/txt/'

"""alex"""
path = '/media/xuus/A45ED35B5ED324B8/LRP-HAM_git/generate_xml_include_domain_label/20211229_add_domain_label/target/gt/'
path_txt = '/media/xuus/A45ED35B5ED324B8/LRP-HAM_git/generate_xml_include_domain_label/20211229_add_domain_label/target/txt/'

if not os.path.exists(path_txt):
    os.mkdir(path_txt)

# def read_file(file_dir):
data_path = []
img_path = os.listdir(path)
img_path = sorted(img_path, key=lambda x: int(x[:-4]))
for item_name in img_path:
    # print(os.path.splitext(item_name))
    data_path.append(path + os.path.splitext(item_name)[0] + ".jpg")

for path in data_path:
    tru = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ret, blu = cv2.threshold(tru, 200, 255, cv2.THRESH_TOZERO)
    edge = cv2.Canny(blu, 50, 150)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        os.remove(path)
        # os.remove("/home/irislab/fastrcnn/喉鏡/會厭/add_black_pic/tru/"+os.path.basename(path))
        print(path + os.path.basename(path) + "   removed! ")
        continue
    if len(contours) != 1:
        print(os.path.basename(path))

    for c in contours:

        x, y, w, v = cv2.boundingRect(c)

        cv2.rectangle(tru, (x, y), (x + w, y + v), (255, 255, 255), 2, 8, 0)
        # cv2.imwrite('/home/iris/latest/訓練資料/網路資料加入測試/網路result/rpn/gt/'+os.path.splitext(path.split("/",)[-1])[0]+'.jpg',tru)
        if x == 0:
            x = 1
            # print(os.path.splitext(path.split("/",)[-1])[0])
        if y == 0:
            y = 1
            # print(os.path.splitext(path.split("/",)[-1])[0])

        with open(path_txt + os.path.splitext(path.split("/", )[-1])[0] + ".txt", "a") as f:
            f.write(str(x) + "," + str(y) + "," + str(x + w) + "," + str(y + v) + ",uvula" +  ',' + domain_label + "\n")
            f.close()


                

