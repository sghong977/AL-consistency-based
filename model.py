import torch
import torch.nn as nn
import torch.nn.functional as F

#### RESNET
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    def get_embedding_dim(self):
        return 512

#### SETTINGS
block = [BasicBlock, Bottleneck] # resnet18, 50
is_resnet50 = 0

num_classes = -1
input_channel = 3
n_blocks = [
    [2,2,2,2],  # resnet18
    [3,4,6,3] # resnet50
]

class ResNet(nn.Module):
    def __init__(self, block=block[is_resnet50], num_blocks=n_blocks[is_resnet50]):
        super(ResNet, self).__init__()
        global n_classes, input_channel

        self.in_planes = 64
        num_classes = n_classes
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) #stl10, stride2. because of the input size
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        _out = out.view(out.size(0), -1)  # get output before FC
        e1 = _out
        out = self.linear(_out)
        return out, e1   #, _out # not consider feature similarity
    
    def get_embedding_dim(self):
        return 512


# UPDATE : learing_loss param (default false) for cifar 10
# for the learning loss approach, network should have loss prediction module.
def get_net(name, model=False):
    global n_classes, input_channel
    n_classes = 10

    #input channel and classes
    if (name == 'MNIST') or (name=='FashionMNIST'):
        input_channel = 1
    elif name == 'CIFAR100':
        n_classes = 100

    # select model (Res18, Res50, LL+Res18, LL+Res50)
    if model == 'resnet50':
        is_resnet50 = 1 # res50
    else:
        is_resnet50 = 0 # res18
    return ResNet
