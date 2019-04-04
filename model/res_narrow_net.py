import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class ResNarrowNet(BaseModel):
    def __init__(self, num_classes=10, n_layers=24, n_feature_maps=19, res_pool=(2, 2), use_dilation=False):
        super(ResNarrowNet, self).__init__()
        self.output_size = num_classes
        self.n_layers = n_layers
        self.n_feature_maps = n_feature_maps

        if isinstance(res_pool, list):
            res_pool = tuple(res_pool)

        self.conv0 = nn.Conv2d(1, self.n_feature_maps, (3, 3), padding=(1, 1), bias=False)
        self.pool = nn.AvgPool2d(res_pool)
        if use_dilation:
            self.convs = [nn.Conv2d(self.n_feature_maps, self.n_feature_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(self.n_layers)]
        else:
            self.convs = [nn.Conv2d(self.n_feature_maps, self.n_feature_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(self.n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(self.n_feature_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        self.fc = nn.Linear(self.n_feature_maps, num_classes)

        self.old_fcs = {}
        self.swap_counter = 0
        self.__set_fc_id__()

    def forward(self, x):
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        x = self.fc(x)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def swap_fc(self, num_classes, fc_id=None):
        self.__store_fcs__()

        if not fc_id:
            # new layers
            self.fc = nn.Linear(self.n_feature_maps, num_classes)
            fc_id = self.__set_fc_id__()
        else:
            self.__load_fcs__(fc_id)

        self.output_size = self.fc.out_features

        return fc_id

    def __set_fc_id__(self):
        fc_id = self.swap_counter
        self.fc.id = fc_id
        self.swap_counter += 1
        return fc_id

    def __store_fcs__(self):
        self.old_fcs[self.fc.id] = {}
        self.old_fcs[self.fc.id]['fc'] = self.fc

    def __load_fcs__(self, fc_id):
        assert self.old_fcs[fc_id]
        self.fc = self.old_fcs[fc_id]['fc']
