import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import DiffusionModel #MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse
from omegaconf import OmegaConf

def create_mnist_dataloaders(batch_size,image_size=28,num_workers=0):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #from the range of [0,1] to [-1,1]

    train_dataset = MNIST(root = "./mnist_data", train = True, download = True, transform = preprocess)
    test_dataset = MNIST(root="./mnist_data", train = False, download=True, transform=preprocess)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

    return (train_loader, test_loader)
            



# def parse_args():
#     parser = argparse.ArgumentParser(description="Training DiffusionModel")
#     parser.add_argument('--lr',type = float ,default=0.001)
#     parser.add_argument('--batch_size',type = int ,default=128)    
#     parser.add_argument('--epochs',type = int,default=100)
#     parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
#     parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
#     parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
#     parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
#     parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
#     parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
#     parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
#     parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
#     parser.add_argument('--cpu',action='store_true',help = 'cpu training')

#     args = parser.parse_args()

#     return args


def main(conf):

    #device = "cpu" if args.cpu else "cuda"
    device = conf.cpu

    train_dataloader,test_dataloader = create_mnist_dataloaders(batch_size = conf.data.train_bs, image_size = 28)
    model = DiffusionModel(timesteps = conf.model.timesteps, # sampling steps of DDPM
                image_size = 28,
                in_channels = 1, # input channel size
                base_dim = conf.model.base_dim, #? 
                dim_mults=[2,4])
    model = model.to(device)

    # EMA setting 
    adjust = 1 * conf.data.train_bs * conf.model.ema_steps / conf.model.epochs
    alpha = 1.0 - conf.model.ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(),lr=conf.model.lr) # adam for weight decay
    scheduler = OneCycleLR(optimizer, conf.model.lr, total_steps = conf.model.epochs * len(train_dataloader), pct_start = 0.25, anneal_strategy = 'cos') # from initial lr, 1 cycle annealing
    loss_fn = nn.MSELoss(reduction='mean') # L2

    # load checkpoint
    if conf.ckpt:
        ckpt=torch.load(conf.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps = 0
    for i in range(conf.model.epochs): # #(epoch)
        model.train() # train mode

        # 🚨 not LDM - input (resized) image itself
        for j, (image, target) in enumerate(train_dataloader):
            noise = torch.randn_like(image).to(device) # noise size is the same to image size
            image = image.to(device)
            pred = model(image, noise) # input : image and noise / output : predicted noise
            
            loss = loss_fn(pred, noise)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            scheduler.step()

            if global_steps % conf.model.ema_steps==0:
                model_ema.update_parameters(model)
            global_steps +=1

            if j % conf.log_freq==0:
         
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,conf.model.epochs,j,len(train_dataloader),
                                                                    loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
        ckpt={"model":model.state_dict(),
                "model_ema":model_ema.state_dict()}

        os.makedirs("results",exist_ok=True)
        torch.save(ckpt,"results/steps_{:0>8}.pt".format(global_steps))

        model_ema.eval()
        samples = model_ema.module.sampling(conf.data.n_samples, clipped_reverse_diffusion = not conf.no_clip, device = device)
        save_image(samples,"results/steps_{:0>8}.png".format(global_steps),nrow=int(math.sqrt(conf.data.n_samples)))

if __name__=="__main__":
    # args=parse_args()
    conf = OmegaConf.load("./configs/train.yaml")
    main(conf)