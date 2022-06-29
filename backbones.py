import torch.nn as nn
from torchvision import models

resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

def get_backbone(name):
    if "resnet" in name.lower():
        return ResNetBackbone(name)
    elif "lenet" == name.lower():
        return LeNetFc()
    elif "alexnet" == name.lower():
        return AlexNetBackbone()
    elif "dann" == name.lower():
        return DaNNBackbone()

class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224*224*3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim

class LeNetFc(nn.Module):

    def __init__(self):
        super(LeNetFc, self).__init__()
        self.features = nn.Sequential()
        self.features.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.features.add_module('f_bn1', nn.BatchNorm2d(64))
        self.features.add_module('f_pool1', nn.MaxPool2d(2))
        self.features.add_module('f_relu1', nn.ReLU(True))
        self.features.add_module('f_conv2', nn.Conv2d(64, 128, kernel_size=5))
        self.features.add_module('f_bn2', nn.BatchNorm2d(128))
        self.features.add_module('f_pool2', nn.MaxPool2d(2))
        self.features.add_module('f_relu2', nn.ReLU(True))

        self.classifier = nn.Sequential()
        self.classifier.add_module('c_fc1', nn.Linear(128 * 5 * 5, 1024))
        self.classifier.add_module('c_relu1', nn.ReLU())
        self.classifier.add_module('c_drop1', nn.Dropout2d(p=0.5))
        self.classifier.add_module('c_fc2', nn.Linear(1024, 256))

        self.__in_features = self.classifier[-1].out_features
    def forward(self, input_data):
        feature = self.features(input_data)

        feature= feature.view(feature.size(0), -1)
        class_output = self.classifier(feature)

        return class_output


    # Exactly like forward function, but return features
    def get_feature(self, input_data):
        feature = self.features(input_data)
        feature = feature.view(-1, 128 * 5 * 5)
        fea = self.classifier.c_fc1(feature)
        fea = self.classifier.c_relu1(fea)
        fea = self.classifier.c_drop1(fea)
        fea = self.classifier.c_fc2(fea)
        fea = self.classifier.c_bn2(fea)
        fea = self.classifier.c_relu2(fea)

        return fea

    def output_num(self):
        return self.__in_features

# convnet without the last layer
class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_alexnet.classifier[i])
        self._feature_dim = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def output_num(self):
        return self._feature_dim


