import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torchstat


class FCNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(FCNet, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(56180, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 56180)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    fcNet = FCNet(100, 10)
    stats = torchstat.stat_simple(fcNet, (100,))
    print("Stats for FCNet:")
    print("total_parameters, total_memory, total_madds, total_flops, total_duration, total_mread, total_mwrite, "
          "total_memrw")
    print(stats)

    # Get stats of self defined network
    model = Net()
    # torchstat.stat(model, (3, 224, 224))
    stats = torchstat.stat_simple(model, (3, 224, 224))
    print("Stats for Net:")
    print("total_parameters, total_memory, total_madds, total_flops, total_duration, total_mread, total_mwrite, "
          "total_memrw")
    print(stats)
    print()

    # Get stats of provided networks

    model = models.resnet50()
    stats = torchstat.stat_simple(model, (3, 224, 224))
    print("Stats for Resnet50:")
    print("total_parameters, total_memory, total_madds, total_flops, total_duration, total_mread, total_mwrite, "
          "total_memrw")
    print(stats)

    model = models.resnet101()
    stats = torchstat.stat_simple(model, (3, 224, 224))
    print("Stats for Resnet101:")
    print("total_parameters, total_memory, total_madds, total_flops, total_duration, total_mread, total_mwrite, "
          "total_memrw")
    print(stats)
