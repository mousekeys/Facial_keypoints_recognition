import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms  
from torch.utils.data import Dataset


class Facekeypoint(Dataset):
    def __init__(self,csv_path,split):
        super(Facekeypoint).__init__()
        self.csv_path=csv_path
        self.split=split
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.df=pd.read_csv(self.csv_path)
        self.normal=transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        img,org_img_size=self.get_img(index)
        keypoints=self.get_keypoints(index,org_img_size)
        return img,keypoints
    
    def get_img(self,index):
        img_path=os.path.join(os.getcwd(),self.split,self.df.iloc[index,0]) #Get image at index and its 0th column
        img=Image.open(img_path).convert('RGB') #Read image turn to RGB
        org_img_size=img.size 

        img=img.resize((224,224)) #Resize image to make it 224x224 for all images
        img=np.array(img)/255.0 #Normalize image
        img=torch.tensor(img).permute(2,0,1).float() # Change structure to match requirement of pytorch
        img=self.normal(img) #Normalize image more

        return img.to(self.device),org_img_size#Send image to GPU or CPU

    def get_keypoints(self,index,org_img_size):
        keypoints=self.df.iloc[index,1:].to_numpy().astype(np.float32)
        kp_x=keypoints[0::2]/org_img_size[0]
        kp_y=keypoints[1::2]/org_img_size[1]
        kp=np.concatenate([kp_x,kp_y])

        return torch.tensor(kp).to(self.device)
    

    def load_img(self,index):
        img_path=os.path.join(os.getcwd(),self.split,self.df.iloc[index,0])
        img=Image.open(img_path).convert('RGB')
        img=img.resize((224,224))
        return np.asarray(img)/255.0
        

if __name__=='__main__':
    train=Facekeypoint(r'data\training_frames_keypoints.csv',r'data\training')
    test=Facekeypoint(r'data\test_frames_keypoints.csv',r'data\test')
    
