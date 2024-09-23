import pandas as pd
import numpy as np
from torchvision import  transforms , models
from torch.utils.data import Dataset , DataLoader 
import torch.nn as nn
import matplotlib.pyplot as plt    
import torch
import os
from tqdm import tqdm
from PIL import Image
import json

#import made files
from src.face_kp.config.config import configuration
from src.face_kp.datasets.datasets import Facekeypoint
from src.face_kp.model.model import get_model
from src.face_kp.utils.utils import train,plot_loss,visualize


def main():
    saved_path=os.path.join(os.getcwd(),'dump',configuration.get('saved_path'))
    model_path=os.path.join(saved_path,'model.pth')
    hyperparam_path=os.path.join(saved_path,'hyperparam.json')

    train_loss_path=os.path.join(saved_path,'train_loss.png')
    outputimg_save_path=os.path.join(saved_path,'image_keypoint.png')

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    traindatset = Facekeypoint(configuration.get('train_csv_pth'),r'data\training')
    testdata=Facekeypoint(configuration.get('test_csv_pth'),r'data\test')

    training_data=DataLoader(traindatset,batch_size=configuration.get('batch_size'),shuffle=True)
    testing_data=DataLoader(testdata,batch_size=configuration.get('batch_size'),shuffle=False)

    vgg16_model=get_model(device=device)
    optimizer=torch.optim.Adam(vgg16_model.parameters(),lr=configuration.get('learning_rate'))
    criterion=nn.L1Loss()


    train_loss,test_loss=train(configuration.get('n_epoch'),training_data=training_data,testing_data=testing_data,vgg16_model=vgg16_model,optimizer=optimizer,criterion=criterion)
    plot_loss(train_loss,test_loss,train_loss_path)
    visualize('fae.img',outputimg_save_path,vgg16_model,configuration.get('model_input_size'),device)


    with open(hyperparam_path,'w') as f:
        json.dump(configuration,f)

    torch.save(vgg16_model,model_path)

if __name__=='__main__':
    main()