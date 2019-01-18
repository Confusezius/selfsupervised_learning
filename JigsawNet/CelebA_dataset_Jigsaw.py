"""========================================================"""
### LIBRARIES
import numpy as np, os, sys, pandas as pd, csv
import torch, torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import os
from tqdm import trange

# source_path = '/export/home/kroth/Project_Manifold/LOADDATA/IMAGES'
# perm_path   = '/export/home/kroth/Project_Manifold/manifoldlearning/Network_Training/JigsawNet/permutations_classes-200_tiles-9.npy'
# source_path = '/home/karsten_dl/Dropbox/Data_Dump/DeepClustering/LOADDATA/IMAGES'
# perm_path   = '/home/karsten_dl/Dropbox/Projects/current_projects/DeepClustering/Network_Training/JigsawNet/permutations_classes-200_tiles-9.npy'



"""========================================================"""
### MISC
def color_jitter(x):
    x = np.array(x, 'int32')
    for ch in range(x.shape[-1]):
        x[:,:,ch] += np.random.randint(-2,2)
    x[x>255] = 255
    x[x<0]   = 0
    return x.astype('uint8')




"""========================================================"""
### DATASET
class CelebA_Dataset(Dataset):
    def __init__(self, opt, mode='train'):
        self.num_tiles, self.num_classes   = opt.num_tiles, opt.num_classes
        self.preload = opt.preload

        source_path, perm_path = opt.data_path, opt.perm_path
        seed, perc_data, tv_split = opt.seed, opt.perc_data, opt.tv_split

        self.images         = [source_path+'/'+x for x in os.listdir(source_path)]
        self.permutations   = np.load(perm_path)
        self.permutations   = self.permutations-self.permutations.min()

        means_CelebA   = np.array([0.5064, 0.4263, 0.3845])
        stds_CelebA    = np.array([0.1490, 0.1443, 0.1469])

        self.adjust_format  = transforms.Compose([transforms.CenterCrop((216,168))])
        # self.adjust_format  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=means_CelebA, std=stds_CelebA)])

        self.tile_size      = (64,48)
        self.augment_tile   = transforms.Compose([transforms.RandomCrop(self.tile_size),
                                                  transforms.Lambda(color_jitter),
                                                  transforms.ToTensor()])


        datalen = int(len(self.images)*perc_data)
        self.images = self.images[:datalen]
        datalen = int(len(self.images)*tv_split)

        if mode=='train':
            self.images  = self.images[:datalen]
        elif mode=='val':
            self.images  = self.images[datalen:]


        self.perc_data   = perc_data
        self.rng         = np.random.RandomState(seed)
        self.n_files     = len(self.images)

        if self.preload:
            self.all_images  = []
            for i in trange(len(self.images), desc='Preloading images to RAM...'):
                self.all_images.append(self.adjust_format(Image.open(self.images[i])))


    def __getitem__(self, idx):
        if self.preload:
            img = self.all_images[idx]
        else:
            img = self.adjust_format(Image.open(self.images[idx]))

        if np.random.rand()<0.3:
            img = img.convert('LA').convert('RGB')

        tile_size_x, tile_size_y = img.size[0]//3, img.size[1]//3
        tiles   = [None]*self.num_tiles


        for i in range(self.num_tiles):
            x,y  = i//3, i%3
            crop = [tile_size_x*x, tile_size_y*y]
            crop = [crop[0], crop[1], crop[0]+tile_size_x, crop[1]+tile_size_y]
            # crop = np.array([crop[0]-a_x, crop[1]-a_y, crop[0]+a_x+1, crop[1]+a_y+1]).astype(int)
            tile = img.crop(crop)
            # from IPython import embed
            # embed()
            tile = self.augment_tile(tile)
            tile_mean, tile_sd = tile.view(3,-1).mean(dim=1).numpy(), tile.view(3,-1).std(dim=1).numpy()
            tile_sd[tile_sd==0] = 1
            norm     = transforms.Normalize(mean=tile_mean.tolist(), std=tile_sd.tolist())
            tile     = norm(tile)
            tiles[i] = tile

        rand_perm = np.random.randint(len(self.permutations))
        tiles = [tiles[self.permutations[rand_perm][i]] for i in range(self.num_tiles)]
        tiles = torch.stack(tiles,dim=0)

        return {'Tiles':tiles, 'Target':int(rand_perm)}

    def __len__(self):
        return self.n_files
