import pickle
file_name = 'aeroplane_pr'
with open('./fr-rcnn-weights/res101/voc_2007_test/default/res101_faster_rcnn_iter_180000/'+file_name+'.pkl', 'rb') as f:
    model = pickle.load(f)
    print(model)# rec prec ap
    print('rec',model['rec'].shape)
    print('prec',model['prec'].shape)
    print('ap',model['ap'].shape)


'''
train_x, train_y = model

import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.imshow(train_x[0].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()
'''
