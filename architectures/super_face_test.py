#!/usr/bin/python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import re
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as f
from PIL import Image
import torch
import sys, os
import random
import time
import numpy as np
import pickle
import torchvision.models as models
from tempfile import TemporaryFile
import csv
import math
import tensorflow as tf
import sys
import image_editor 
ngf=32

list_attr=['5_o_Clock_Shadow','Arched_Eyebrows', 'Bags_Under_Eyes', 'Big_Lips', 'Big_Nose','Bushy_Eyebrows', 
'Double_Chin', 'Goatee','Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 
'Mustache', 'Narrow_Eyes','No_Beard', 'Pointy_Nose', 'Sideburns',  'Young']

nAttr=18
imageSize_LR=16
imageSize_HR=128
batch_num=16

def get_loader(image_dir, image_size, 
               batch_size, mode='train', num_workers=0):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(224)) #IMAGE SIZE
    transform.append(T.Resize((image_size)))
    transform.append(T.ToTensor())
    #transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader

#Path to ./DataFolder/Fams/
#Loads real images
HR_loader = get_loader('C:/Users/Yami/Desktop/CMSC498V/Image-Repair/architectures/DataFolder/',image_size=((224,224)),batch_size=batch_num,mode='train',num_workers=0)
LR_loader = get_loader('C:/Users/Yami/Desktop/CMSC498V/Image-Repair/architectures/DataFolder/',image_size=((224,224)),batch_size=batch_num,mode='train',num_workers=0)
class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        ##(W-F+2P)/S  + 1    Output size of conv layer
        #W = width, F = filter size, S = stride, P = padding
        self.conv1 = nn.Conv2d(3, 32, 5, padding=0, bias = False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 5, padding=0, bias = False)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, 4, padding=0, bias = False)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, 4, padding=0, bias = False)
        self.relu4 = nn.ReLU()
        self.fc1   = nn.Linear(11*11*8*32,1024)
        self.relu5 = nn.ReLU()
        self.fc2   = nn.Linear(1024,1)
        self.sig   = nn.Sigmoid()
		
	
    def forward(self, x):
        #print(x.shape)
        x=self.relu1(F.avg_pool2d(self.conv1(x),(2,2))) #(224-4)/2 = 110
        #print(x.shape)
        x=self.relu2(F.avg_pool2d(self.conv2(x),(2,2))) #(110-4)/2 = 53
        #print(x.shape)
        #print("y.shape")
        #print(y.shape)
        x=self.relu3(F.avg_pool2d(self.conv3(x),(2,2))) #(53-3)/2 = 25
        x=self.relu4(F.avg_pool2d(self.conv4(x),(2,2))) #(25-3)/2 = 11
        x=x.view(-1,11*11*8*32)
        x=self.relu5(self.fc1(x))
        x=self.sig(self.fc2(x))
        return x
				
netD = D()

class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        # #encoding
        # self.conv1 = nn.Conv2d(3,ngf,4,stride=2,padding=1,bias=False)
        # self.batch_norm1 = nn.BatchNorm2d(ngf)
        # self.leaky_relu1 = nn.LeakyReLU(.2,inplace=True)
        
        # self.conv2 = nn.Conv2d(ngf,4*ngf,4,stride=2,padding=1,bias=False)
        # self.batch_norm2 = nn.BatchNorm2d(4*ngf)
        # self.leaky_relu2 = nn.LeakyReLU(.2,inplace=True)
        
        # self.conv3 = nn.Conv2d(4*ngf,16*ngf,4,stride=2,padding=1,bias=False)
        # self.batch_norm3 = nn.BatchNorm2d(16*ngf)
        # self.leaky_relu3 = nn.LeakyReLU(.2,inplace=True)
        
        # self.conv4 = nn.Conv2d(16*ngf,64*ngf,2, bias=False)
        # self.batch_norm4 = nn.BatchNorm2d(64*ngf)
        # self.leaky_relu4 = nn.LeakyReLU(.2,inplace=True)
        
        # #decoding
        # self.decode1 = nn.ConvTranspose2d(ngf*64+nAttr,ngf*32,2,bias=False)
        # self.batch_norm5 = nn.BatchNorm2d(32*ngf)
        # self.relu1 = nn.ReLU()
        
        # self.decode2 = nn.ConvTranspose2d(48*ngf,ngf*24,4,stride=2,padding=1,bias=False)
        # self.batch_norm6 = nn.BatchNorm2d(24*ngf)
        # self.relu2 = nn.ReLU()
        
        # self.decode3 = nn.ConvTranspose2d(28*ngf,16*ngf,4,stride=2,padding=1,bias=False)
        # self.batch_norm7 = nn.BatchNorm2d(16*ngf)
        # self.relu3 = nn.ReLU()
        
        # self.decode4 = nn.ConvTranspose2d(17*ngf,8*ngf,4,stride=2,padding=1, bias=False)
        # self.batch_norm8 = nn.BatchNorm2d(8*ngf)
        # self.relu4 = nn.ReLU()
        
        self.deconv1 = nn.ConvTranspose2d(3,4*ngf, 3,stride=1,padding=1, bias = False) #(224-2+2) = 224
        self.batch_norm9 = nn.BatchNorm2d(4*ngf)
        
        self.deconv2 = nn.ConvTranspose2d(4*ngf,2*ngf, 3,stride=1, padding=1, bias = False) 
        self.batch_norm10 = nn.BatchNorm2d(2*ngf)
        
        self.deconv3 = nn.ConvTranspose2d(2*ngf,ngf,3,stride=1, padding=1, bias = False)
        self.batch_norm11 = nn.BatchNorm2d(ngf)
        
        self.deconv4 = nn.ConvTranspose2d(ngf,3,5,stride=1,padding=2, bias=False)
		
    def forward(self, d2):
        d2 = self.batch_norm9(self.deconv1(d2))
        d2 = self.batch_norm10(self.deconv2(d2))
        d2 = self.batch_norm11(self.deconv3(d2))
        d2 = self.deconv4(d2)
        return d2
		


netG = G()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    netD = nn.DataParallel(netD,device_ids=[0,1])
    netG = nn.DataParallel(netG,device_ids=[0,1])
	
if torch.cuda.device_count() == 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    netD = nn.DataParallel(netD,device_ids=[0])
    netG = nn.DataParallel(netG,device_ids=[0])

#Loading weights
#netD.load_state_dict(torch.load('C:/Users/Yami/Desktop/CMSC498V/Image-Repair/models/face_PL_D.pth')) 
#netG.load_state_dict(torch.load('C:/Users/Yami/Desktop/CMSC498V/Image-Repair/models/face_PL_G.pth'))

netG = netG.to(device)
netD = netD.to(device)
#torch.cuda.synchronize()

criterion_D = nn.BCELoss() #Binary cross entorpy lost 
criterion_G = nn.BCELoss() 
criterion_LH = nn.L1Loss()
criterion_LV = nn.L1Loss()

optimizerD = optim.RMSprop(netD.parameters(),lr=0.001,alpha=.9)
optimizerG = optim.RMSprop(netG.parameters(),lr=0.001,alpha=.9)

#Decreasing learning rate
# schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=1, gamma=0.95)
# schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=1, gamma=0.95)

                            
scale_LR = T.Compose([T.ToPILImage(),
                            T.Resize((224,224),interpolation=0),
                            T.ToTensor(),
                            T.Normalize(mean = [0.5, 0.5, 0.5],std = [0.5, 0.5, 0.5])
                            ])
#Need a function that takes an image and add noise 
toImage = T.Compose([T.ToPILImage()])
toTensor = T.Compose([T.ToTensor()])

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
total_var_loss = TVLoss()

#Training phase
#Number of epochs 
for epoch in range(0,100000):
    # if(epoch<135):
    #     for g in optimizerD.param_groups:
    #         g['lr'] = 0.001*math.pow(.95,epoch)
    #     for g in optimizerG.param_groups:
    #         g['lr'] = 0.001*math.pow(.95,epoch)   
    # if(epoch<100):
    #     for g in optimizerD.param_groups:
    #         g['lr'] = 0.001*.2*math.pow(.95,epoch)
    #     for g in optimizerG.param_groups:
    #         g['lr'] = 0.001*.2*math.pow(.95,epoch)
		
    if(epoch<50):
        for g in optimizerD.param_groups:
            g['lr'] = 0.001*.02*math.pow(.95,epoch)
        for g in optimizerG.param_groups:
            g['lr'] = 0.001*.02*math.pow(.95,epoch)
					
    for i, data_HR in enumerate(HR_loader,0):
        torch.cuda.empty_cache()
        netD.zero_grad()
        start=time.time()
       

        real,attr = data_HR

        LR_input= torch.FloatTensor(batch_num,3,224,224) #LR is the noise image
        # #Apply noise to image
        # #real is ground truth
        # #LR is noisy image
        mask_input = torch.FloatTensor(batch_num,3,224,224)
        opposite_mask_input = torch.FloatTensor(batch_num,3,224,224)
        for a in range(0,len(real)):
            w = toImage(real[a]) 
            w,mask = image_editor.sample_damaging(w)
            mask_input[a] = toTensor(mask)
            mask = np.array(mask)
            opposite_mask = np.where(mask > .5,0,1)
            opposite_mask = (opposite_mask*255).astype(np.uint8)
            opposite_mask = Image.fromarray(opposite_mask)
            opposite_mask_input[a] = toTensor(opposite_mask)
            LR_input[a] = toTensor(w)
            #LR_input[a] = scale_LR(real[a]) 

        real = real.to(device)
        real = Variable(real)

        #Discrimator update
        target = Variable(torch.ones(real.size()[0])).to(device)

        output = netD(real)

        errD_real = criterion_D(output, target) #errD is the discrimator loss on idenifying real images
        LR_input = LR_input.to(device)
        fake=netG(LR_input)
        target = Variable(torch.zeros(real.size()[0])).to(device)
        output = netD(fake.detach())

        errD_fake = criterion_D(output, target) #errD_fake is the discrimator loss on idenifying fake images
        #print(output)
        errD = (errD_real + errD_fake)
        errD.backward()
        #print(errD)
        optimizerD.step()
        
        #Generator update
        netG.zero_grad()
        target = Variable(torch.ones(real.size()[0])).to(device)
        output = netD(fake)
        errG_V =total_var_loss(fake.to(device)) 

        #turn mask to numpy array
        #turn fake to numpy array
        #copy real, and turn to numpy array
        copy_fake = fake
        copy_real = real
        copy_mask = mask_input
        FV = torch.FloatTensor(batch_num,3,224,224)
        RV = torch.FloatTensor(batch_num,3,224,224)

        for a in range(0,len(real)):

            current_real = copy_real[a].cpu().detach().numpy()
            current_fake = copy_fake[a].cpu().detach().numpy()
            current_mask = copy_mask[a].numpy()
            FV[a] = torch.from_numpy(np.multiply(current_mask,current_fake))
            RV[a] = torch.from_numpy(np.multiply(current_mask,current_real))
            # zz
            # FV = np.mutiply(mask,copy_fake) #Fake valid
            # RV = np.mutiply(mask,copy_real) #Real valid


        #turn mask to numpy array
        #turn fake to numpy array
        #copy real, and turn to numpy array
        copy_mask = opposite_mask_input
        FH = torch.FloatTensor(batch_num,3,224,224)
        RH = torch.FloatTensor(batch_num,3,224,224)
        for a in range(0,len(real)):

            current_real = copy_real[a].cpu().detach().numpy()
            current_fake = copy_fake[a].cpu().detach().numpy()
            oppsite_mask = copy_mask[a].numpy()
            FH[a] = torch.from_numpy(np.multiply(oppsite_mask,current_fake))
            RH[a] = torch.from_numpy(np.multiply(oppsite_mask,current_real))

        # FH = np.mutiply(oppsite_mask,copy_fake) #Fake valid
        # RH = np.mutiply(oppsite_mask,copy_real) #Real valid

        errG_LV = criterion_LV(FV.to(device),RV.to(device)) 
        errG_LH = criterion_LH(FH.to(device),RH.to(device)) 

		
        if i % 25 ==0:
            #Change the paths
            vutils.save_image(real, '%s/real_PL.png' % "C:/Users/Yami/Desktop/CMSC498V/Image-Repair/results/", normalize = False)
            save_fake = netG(LR_input)
            vutils.save_image(save_fake.data, '%s/fake_PL_epoch_%03d.png' % ("C:/Users/Yami/Desktop/CMSC498V/Image-Repair/results/", epoch), normalize = False)

        

        del fake
        del real
        del LR_input
		
        errG_G = criterion_G(output, torch.t(target.unsqueeze(0))) #May not need to do transpose
        #output = torch.log(output)
        #output = torch.mean(output)

        #errG = (max(math.pow(.995,epoch)*.01,.005)*errG_G.to(device)) + errG_L.to(device)
        errG = (.01*errG_G.to(device)) + 6*errG_LH + errG_LV + .1*errG_V
        errG.backward()
        optimizerG.step()
        end=time.time()
        

        if i % 10 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Time Elapsed:%.4f s' % (epoch, 135, i, len(HR_loader), errD.data, errG.data,(end-start)))
            #print(torch.cuda.max_memory_allocated(device))

    if(epoch<100):
        for g in optimizerD.param_groups:
            g['lr'] = 0.001*math.pow(.95,epoch)
        for g in optimizerG.param_groups:
            g['lr'] = 0.001*math.pow(.95,epoch)
    
    #Change Path			
    torch.save(netG.state_dict(),'C:/Users/Yami/Desktop/CMSC498V/Image-Repair/models/face_PL_G.pth')
    torch.save(netD.state_dict(),'C:/Users/Yami/Desktop/CMSC498V/Image-Repair/models/face_PL_D.pth')		
