from gym import spaces
import torch.nn as nn
import torch.nn.functional as F
import torch

# Class structure loosely inspired by https://towardsdatascience.com/beating-video-games-with-deep-q-networks-7f73320b9592
class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    neurips DQN paper.
    """

    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert type(
            observation_space) == spaces.Box, 'observation_space must be of type Box'
        assert len(
            observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        assert type(
            action_space) == spaces.Discrete, 'action_space must be of type Discrete'
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[0], out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=32*9*9 , out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=action_space.n)
        )

        self.hx = nn.Parameter(torch.FloatTensor(1, 512), requires_grad=False)
        self.Wa = nn.Linear(2592, 512)
        self.Wh = nn.Linear(512, 512)
        self.att = nn.Linear(512, 2592)
        self.a = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.g = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha_prev = nn.Parameter(torch.FloatTensor(1, 2592), requires_grad=False)
        self.hx_prev = nn.Parameter(torch.FloatTensor(1, 512), requires_grad=False)

        # init
        self.g.data.fill_(1)
        self.a.data.fill_(1)
        self.alpha_prev.data.fill_(0)
        self.hx_prev.data.fill_(0)

        self.train()

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0],-1)        # torch.Size([1, 2592])
        # print("out", conv_out.shape)
        x = conv_out                    # torch.Size([1, 2592])
        Wa = self.Wa(x)                 # [1, 512]
        Wh = self.Wh(self.hx_prev)           # [1, 512]

        W_a_h = Wa + Wh

        k = torch.tanh(self.a * W_a_h)
        at = F.log_softmax(self.att(k), dim=1)      # [1, 2592]

        alpha_ = at * self.g + self.alpha_prev * (1 - self.g)

        # print("alpha", alpha_.shape)
        # update self.alpha_prev
        if alpha_.shape == self.alpha_prev.shape:
            self.alpha_prev.data = alpha_

        Yt = F.log_softmax(alpha_, dim=1) * x       # [1, 2592]

        r = torch.sigmoid(W_a_h)                    # [1, 512]
        z = torch.sigmoid(W_a_h)

        h_ = torch.tanh((r * Wh) + Wa) * (1 - z)
        h = self.hx_prev * z                             # [1, 512]

        ht = h + h_
        # update self.hx_prev
        if ht.shape == self.hx_prev.shape:
            self.hx_prev.data = ht

        fc_out = self.fc(Yt)
        # print("hx", self.hx.shape)
        # print("fc_out", fc_out.shape)
        return fc_out
