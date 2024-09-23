import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

def train_batch(img,kps,vgg16_model,optimizer,criterion):
    vgg16_model.train()
    optimizer.zero_grad()

    #forward_pass
    pred=vgg16_model(img)
    loss=criterion(pred,kps)

    #backward pass
    loss.backward()
    optimizer.step()

    return loss

@torch.no_grad()
def model_test(img,kps,vgg16_model,criterion):
    vgg16_model.eval()
    pred=vgg16_model(img)
    loss=criterion(pred,kps)

    return loss


def train(n_epoch,training_data,testing_data,vgg16_model,optimizer,criterion):
    train_loss=[]
    test_loss=[]

    for epoch in range(1,n_epoch+1):
        epoch_train_loss,epoch_test_loss=0,0


        for img,kps in tqdm(training_data,desc=f'Training {epoch} of {n_epoch}'):
            loss=train_batch(img,kps,vgg16_model,optimizer,criterion)
            epoch_train_loss+=loss.item()
        epoch_train_loss/=len(training_data)
        train_loss.append(epoch_train_loss)


        for img,kps in tqdm(testing_data,desc=f'Testing {epoch} of {n_epoch}'):
            loss=model_test(img,kps,vgg16_model,criterion)
            epoch_test_loss+=loss.item()
        epoch_test_loss/=len(testing_data)
        test_loss.append(epoch_test_loss)




        print(f'Epoch {epoch}/{n_epoch} : Train Loss : {epoch_train_loss} Test Loss : {epoch_test_loss}')

        return train_loss,test_loss

def plot_loss(train_loss,test_loss,save_path):
    epochs = np.arange(len(train_loss)+1)

    plt.figure()
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, test_loss, 'r', label='Test Loss')
    plt.title("Training and Test Loss Curve Over Epochs")
    plt.xlabel('Epochs')
    plt.ylabel('L1 Loss')
    plt.legend()
    plt.savefig(save_path)


def load_img(img_path,model_input_size,device):
    normal=transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    image=Image.open(img_path).convert('RGB')
    org_img_size=image.size

    image=image.resize((model_input_size,model_input_size))
    image=org_img=np.array(image)/255.0
    image=torch.tensor(image).permute(2,0,1).float()
    image=normal(image)
    return image.to(device),org_img

def visualize(img_path,saved_model_path,model,model_input_size=224,device='cpu'):
    img_preprocess,img = load_img(img_path,model_input_size,device)

    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(img)

    plt.subplot(122)
    plt.title("Image with Facial Keypoints")
    plt.imshow(img)


    kp_s = model(img_preprocess[None]).flatten().detach().cpu()
    kp_s = kp_s.to('cpu')
    plt.scatter(kp_s[:68] * model_input_size, kp_s[68:] * model_input_size, c='y',s = 2)

    plt.savefig(saved_model_path)