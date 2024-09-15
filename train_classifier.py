from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.datasets as datasets

import resnet_models


backbone_weights = models.ResNet152_Weights.IMAGENET1K_V1

resnet152_ensemble = resnet_models.resnet152_ensemble(num_classes=10)
net = resnet152_ensemble

batch_size = 4
num_workers = 4

# Load pre-trained parameters
resnet_models.set_resnet_weights(net, backbone_weights)
resnet_models.freeze_backbone(net)

net = resnet152_ensemble
ensemble_size = len(net.fc_layers)


cifar10_train = datasets.CIFAR10(root="./data/", train=True, transform=resnet_models.preprocess, download=True)
trainloader = torch.utils.data.DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

cifar10_test = datasets.CIFAR10(root="./data/", train=False, transform=resnet_models.preprocess, download=True)
testloader = torch.utils.data.DataLoader(cifar10_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


