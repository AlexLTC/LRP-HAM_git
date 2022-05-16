**gallon code 中主要是將圖像轉為放入網路時所讀取的資料型態(.xml)

reiszeImg_for_ARM
(若要在 DA-FRCNN 加上 attention module，必須將 dataset 的 image 與 gt resize 成一致大小 (640*480)
在 uvula 參雜了 1280*720 與 720*480 的圖像)



produce_txt -> txt2xml 

(將 ground_truth 轉成txt再轉成xml檔，得到位置的訊息)


-> main3files
(切割 training_set, val_set,  test_set)
