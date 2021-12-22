from gym import spaces
import torch.nn as nn
import torch.nn.functional as F
import torch
import tfpyth
import tensorflow as tf

import numpy as np



# Class structure loosely inspired by https://towardsdatascience.com/beating-video-games-with-deep-q-networks-7f73320b9592
class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    neurips DQN paper.
    """

    def __init__(self):#,
                 # observation_space: spaces.Box,
                 # action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()

        # assert type(
        #     observation_space) == spaces.Box, 'observation_space must be of type Box'
        # assert len(
        #     observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        # assert type(
        #     action_space) == spaces.Discrete, 'action_space must be of type Discrete'

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=7, stride=1,padding=3),
            # nn.Conv2d(in_channels=observation_space.shape[0], out_channels=16, kernel_size=8, stride=4),
            # nn.Conv2d的功能是：對由多個輸入平面組成的輸入訊號進行二維卷積，以最簡單的例子進行說明：
            # 輸入訊號的形式為(N, Cin, H, W)，N表示batch size，Cin表示channel個數，H，W分別表示特徵圖的高和寬。
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=256 , out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=300)
            # nn.Linear(in_features=256, out_features=action_space.n)
        )

        # self.hx = nn.Parameter(torch.FloatTensor(1, 300), requires_grad=False)
        self.Wa = nn.Linear(256, 300)
        self.Wh = nn.Linear(300, 300)
        self.att = nn.Linear(300, 256)
        self.a = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.g = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha_prev = nn.Parameter(torch.FloatTensor(1, 38, 38, 256), requires_grad=False)
        self.hx_prev = nn.Parameter(torch.FloatTensor(1, 38, 38, 300), requires_grad=False)

        # init
        self.g.data.fill_(1)
        self.a.data.fill_(1)
        self.alpha_prev.data.fill_(0)
        self.hx_prev.data.fill_(0)

        self.train()

        self.Yt_b = nn.Parameter(torch.FloatTensor(1, 38, 38, 256), requires_grad=False)
        self.ht = nn.Parameter(torch.FloatTensor(1, 38, 38, 300), requires_grad=False)



    def forward(self,x ,hx_prev,key):
        # print('1')
        # x1 = nn.Parameter(torch.FloatTensor(1,2,84,84), requires_grad=False)
        # x = torch.empty(1,2,84,84)
        # x1 = torch.randn(1,2,84,84)
        # x = x1
        conv_out = self.conv(x).view(x.size()[0],-1)        # torch.Size([1, 2592])
        x = torch.reshape(conv_out, (1, 38, 38, 256))

        # print('2')

        # print("out", conv_out.shape)
        # x = conv_out                    # torch.Size([1, 2592])
        # print('3')

        Wa = self.Wa(x)                 # [1, 512]
        Wh = self.Wh(hx_prev)           # [1, 512]

        W_a_h = Wa + Wh

        k = torch.tanh(self.a * W_a_h)
        at = F.log_softmax(self.att(k), dim=3)      # [1, 2592]

        alpha_ = at * self.g + self.alpha_prev * (1 - self.g)

        # print("alpha", alpha_.shape)
        # update self.alpha_prev
        if alpha_.shape == self.alpha_prev.shape:
            self.alpha_prev.data = alpha_

        Yt_b = F.log_softmax(alpha_, dim=3)        # [1, 2592]
        Yt = Yt_b * x

        r = torch.sigmoid(W_a_h)                    # [1, 512]
        z = torch.sigmoid(W_a_h)

        h_ = torch.tanh((r * Wh) + Wa) * (1 - z)
        h = hx_prev * z                             # [1, 512]

        ht = h + h_
        # update self.hx_prev
        # if ht.shape == self.hx_prev.shape:
        #     self.hx_prev.data = ht

        fc_out = self.fc(Yt)

        # print("hx", self.hx.shape)
        # print("fc_out", fc_out.shape)
        # key = 1
        Yt_b = torch.sum(Yt_b, 3)

        if key == 1:
            return Yt_b
        if key == 2:
            return ht
        if key == 3:
            return fc_out
        # return Yt_b, ht, fc_out
    #     self.Yt_b = Yt_b
    #     self.ht = ht
    # def return_value(self, key):
    #     if key == 1:
    #         return self.Yt_b
    #     if key == 2:
    #         return self.ht

DQN = DQN()
def get_tf_output(input_tf,h):
    one = tf.constant(1, tf.float32)
    two = tf.constant(2, tf.float32)
    three = tf.constant(3, tf.float32)

    input_tf = tf.reshape(input_tf, [1, 256, 38, 38])
    # f, ht, yt = tfpyth.tensorflow_from_torch(DQN.forward, [input_tf, h, one], [tf.float32, tf.float32, tf.float32], name=None)
    f = tfpyth.tensorflow_from_torch(DQN.forward, [input_tf, h, one], tf.float32, name=None)
    ht = tfpyth.tensorflow_from_torch(DQN.forward, [input_tf, h, two], tf.float32, name=None)
    yt = tfpyth.tensorflow_from_torch(DQN.forward, [input_tf, h, three], tf.float32, name=None)

    # tfpyth.tensorflow_from_torch(DQN.forward, [input_tf, h], tf.float32, name=None)
    # f = tfpyth.tensorflow_from_torch(DQN.return_value, [one], tf.float32, name=None)
    # ht = tfpyth.tensorflow_from_torch(DQN.return_value, [two], tf.float32, name=None)

    # DQN.forward(input_tf, h)
    # f, ht = tfpyth.tensorflow_from_torch(DQN.return_value, name=None)

    return f, ht, yt


# x = torch.Tensor(5, 3)  # construct a 5x3 matrix, uninitialized

# # input = torch.randn([1,3,84,84])
# input_tf = tf.random_normal([1, 84, 84, 256])
# input_h = tf.random_normal([1, 84, 84, 300])
#
# f, ht = get_tf_output(input_tf, input_h)
# sess = tf.Session()
# print(sess.run(f))
# print(sess.run(ht))

# print(f.shape)
# print(ht.shape)

# x = torch.randn(1,84,84,256)
# # x = torch.randn(3,2592)
#
# Wa = nn.Linear(256, 2)
# Wa = Wa(x)
# print(Wa.size())