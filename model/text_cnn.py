import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


import torch
import torch.nn as nn
from base import BaseModel

class TextConvNet(BaseModel):

    def __init__(self, num_classes=2):

        super(TextConvNet, self).__init__()

        self.embed = nn.Embedding(5000, 300)
        self.conv = nn.ModuleList([nn.Conv2d(1, 32, kernel_size=(k, 300)) for k in [3,4,5]])
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(3*32, num_classes)

    def conv_and_pool(self, x, conv):

        x = F.elu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, x):

        x = self.embed(x)
        x = x.unsqueeze(1)
        x = [F.elu(conv(x)).squeeze(3) for conv in self.conv]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        out = self.fc(x)

        return out

