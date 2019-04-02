import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class LeNet(BaseModel):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.output_size = num_classes
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(80, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.old_fcs = {}
        self.swap_counter = 0
        self.__set_fc_id__()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 80)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.sigmoid(x)
        # return F.softmax(x)
        return x

    def freeze(self):
        for param in self.conv1.parameters():
            param.requires_grad = False

        for param in self.conv2.parameters():
            param.requires_grad = False

        for param in self.fc1.parameters():
            param.requires_grad = False

    def swap_fc(self, num_classes, fc_id=None):
        self.__store_fcs__()

        if not fc_id:
            # new layers
            self.fc2 = nn.Linear(50, num_classes)
            fc_id = self.__set_fc_id__()
        else:
            self.__load_fcs__(fc_id)

        self.output_size = self.fc2.out_features

        return fc_id

    def __set_fc_id__(self):
        fc_id = self.swap_counter
        self.fc2.id = fc_id
        self.swap_counter += 1
        return fc_id


    def __store_fcs__(self):
        self.old_fcs[self.fc2.id] = {}
        self.old_fcs[self.fc2.id]['fc2'] = self.fc2

    def __load_fcs__(self, fc_id):
        assert self.old_fcs[fc_id]
        self.fc2 = self.old_fcs[fc_id]['fc2']
