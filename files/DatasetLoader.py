import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import torchvision.transforms as tf
import matplotlib.pyplot as plt

class AddGaussianNoise():
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


#load data from a folder
class DatasetLoader(Dataset):
    def __init__(self, gray_dir, gt_dir, pytorch=True):
        super().__init__()

        # Loop through the files in red folder and combine, into a dictionary, the other bands
        self.files = [self.combine_files(f, gt_dir) for f in gray_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch
        self.do_augment = False
        self.do_blur    = False
        self.do_resample = True


        self.trsf = tf.Compose([
                tf.RandomAffine(degrees = (-180,180), translate = (0.2, 0.2), scale = (0.5, 1.2)),
                tf.RandomVerticalFlip()
                ])

        self.Blur = tf.GaussianBlur(5,sigma=(0.1,2.0))
        self.Noise = AddGaussianNoise(std=0.05)

    def combine_files(self, gray_file: Path, gt_dir):
        
        files = {'gray': gray_file, 
                 'gt': gt_dir/gray_file.name.replace('gray', 'gt')}

        return files
                                       
    def __len__(self):
        #legth of all files to be loaded
        return len(self.files)
     
    def open_as_array(self, idx, invert=False):
        #open ultrasound data
        raw_us = np.stack([np.array(Image.open(self.files[idx]['gray'])),
                           ], axis=2)
    
        if invert:
            raw_us = raw_us.transpose((2,0,1))
    
        # normalize
        return (raw_us / np.iinfo(raw_us.dtype).max)
    

    def open_mask(self, idx, add_dims=False):
        #open mask file
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        # raw_mask = np.where(raw_mask>100, 1, 0)
        
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx, size = 384):
        #get the image and mask as arrays
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.float32)
        x = F.interpolate(x[None], size = [size, size])[0]

        if self.do_resample:
            y = F.interpolate(y[None, None], size = [size, size])[0,0].long()


        if self.do_augment:
            # do the random movement/scaling of image
            both = self.trsf(torch.cat((x,y[None])))
            x = both[:-1]; y = both[-1].long()

        if self.do_blur:
            # either add any noice/blur to x or do nothing
            choice = np.random.choice(3)
            if choice == 0:
                pass #donothing
            elif choice == 1:
                x = self.Noise(x)
            elif choice == 3:
                x = self.Blur(x)
        return x, y
    
    def get_as_pil(self, idx):
        #get an image for visualization
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8), 'RGB')
    
