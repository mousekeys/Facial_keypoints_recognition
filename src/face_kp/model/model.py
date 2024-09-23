from torchvision import models
import torch.nn as nn
import torch

vgg16_model=models.vgg16(pretrained=True)
vgg16_model.eval()

def get_model(device):
    vgg16_model=models.vgg16(pretrained=True)
    vgg16_model.eval()
    for layers in vgg16_model.parameters():
        layers.requires_grad=False
    vgg16_model.avgpool=nn.Sequential(
        nn.Conv2d(512,512,3),
        nn.MaxPool2d(2),
        nn.Flatten()
    )
    vgg16_model.classifier=nn.Sequential(
        nn.Linear(2048,512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512,136),
        nn.Sigmoid()
    )
    return vgg16_model.to(device=device)


if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=get_model(device)
    print(model)