import torch.nn as nn

class RegresionModel(nn.Module):
    def __init__(self):
        super(RegresionModel, self).__init__()
        self.layer1 = nn.Linear(10, 128)
        self.act_1 = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(128)
        self.d1 = nn.Dropout(0.3)

        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.d2 = nn.Dropout(0.3)
        self.act_2 = nn.ReLU()

        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.d3 = nn.Dropout(0.3)

        self.act_3 = nn.ReLU()
        self.layer4 = nn.Linear(32, 1)

    def forward(self, x):
        activation1 = self.act_1(self.layer1(x))
        activation1 = self.bn1(activation1)
        activation1 = self.d1(activation1)

        activation2 = self.act_2(self.layer2(activation1))
        activation2 = self.bn2(activation2)
        activation2 = self.d2(activation2)

        activation3 = self.act_3(self.layer3(activation2))
        activation3 = self.bn3(activation3)
        activation3 = self.d3(activation3)

        output = self.layer4(activation3)
        return output