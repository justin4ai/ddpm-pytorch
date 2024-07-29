import torch.nn as nn
import torch
import math
from unet import Unet
from tqdm import tqdm

class DiffusionModel(nn.Module):
    def __init__(self,image_size,in_channels,time_embedding_dim=256,timesteps=1000,base_dim=32,dim_mults= [1, 2, 4, 8]):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size

        betas = self._cosine_variance_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        ## register_buffer method sets non-learnable tensors ##
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)) # coefficient of x in DDPM loss
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod)) # coefficient of epsilon in DDPM loss

        # Unet input : x_t and t / output : noise of t-th timestep
        self.model = Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults)

    def forward(self, x, noise):
        # x:NCHW
        t = torch.randint(0, self.timesteps,(x.shape[0],)).to(x.device) # #(batch) times randomly sampled from [0, 999] which is mostly applied in many other implementations

        x_t = self._forward_diffusion(x, t, noise) # for each datapoint in batch, forward-diffused to create x_t

        pred_noise = self.model(x_t, t) # given timestep as a condition, predict ground-truth noise that had been added to form x_t 

        return pred_noise

    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion = True, device = "cuda"):
        x_t = torch.randn((n_samples, self.in_channels, self.image_size, self.image_size)).to(device) # init with noise
        
        for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling"): # #(timesteps) times reverse process
            noise = torch.randn_like(x_t).to(device) # different noise for each reverse step
            t = torch.tensor([i for _ in range(n_samples)]).to(device)

            if clipped_reverse_diffusion: # True in this implementation
                x_t = self._reverse_diffusion_with_clip(x_t, t, noise) # one-step denoised from previous x_t (towards x_0)!

            else:
                x_t=self._reverse_diffusion(x_t, t, noise)


        x_t = (x_t + 1.) / 2. #denormalize : [-1,1] to [0,1]


        return x_t
    
    def _cosine_variance_schedule(self,timesteps,epsilon = 0.008):
        steps=torch.linspace(0, timesteps,steps = timesteps+1, dtype = torch.float32) # timesteps(=1000)-length linspace,
        
        ## According to the paper 'Improved DDPM(p.4)' ##
        f_t=torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5)**2 # range : [0, 1]
        betas = torch.clip(1.0 - f_t[1:] / f_t[ :timesteps], 0.0, 0.999)
        #################################################                                
        return betas

    def _forward_diffusion(self, x_0, t, noise):
        assert x_0.shape == noise.shape

        #q(x_{t}|x_{0})
        first_term = self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * x_0 # indexing t-th sqrt_alphas_cumprod(= coefficient of x in DDPM loss)
        second_term = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0],1, 1, 1) * noise # indexing t-th sqrt_one_minus_alphas_cumprod(= coefficient of eps in DDPM loss)
        
        return first_term + second_term # please refer to DDPM paper, page 4.
                


    ###################################################################################################################
    #################### _reverse .. exactly follows the DDPM paper (page 4, Algorithm2 Sampling method) ##############
    ###################################################################################################################


    @torch.no_grad()
    def _reverse_diffusion(self,x_t,t,noise):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        pred = self.model(x_t,t) # predicted noise of t-th timestep

        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        
        mean = (1. / torch.sqrt(alpha_t)) * (x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred)

        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t-1).reshape(x_t.shape[0], 1, 1, 1)
            std = torch.sqrt(beta_t * (1. -alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            std = 0.0

        return mean + std * noise 


    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t, t, noise): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})-> mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred = self.model(x_t, t) # predicted noise of t-th timestep

        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        
        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred.clamp_(-1., 1.)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                 ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0

        return mean+std*noise 
    