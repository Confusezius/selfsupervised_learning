import os, sys, numpy as np
from tqdm import tqdm, trange
import torch, torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms


import os, sys, numpy as np
from tqdm import tqdm, trange
import torch, torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms



#####################################################################################################################################
class CelebA_Dataset(Dataset):
    def __init__(self, source_path, perc_data=0.5, seed=1, random_sample=True,preload_images=False, mode='train'):
        self.image_sourcepath  = source_path+'/IMAGES'
        self.preload_images    = preload_images

        images_dataframe       = pd.read_table(source_path+'/feature_targets.txt', header=1, delim_whitespace=True)
        self.classnames, self.imagenames = images_dataframe.columns, np.array(images_dataframe.index)
        images_dataframe       = np.array(images_dataframe)

        datalen = int(len(self.imagenames)*perc_data)
        self.imagenames = self.imagenames[:datalen]
        images_dataframe= images_dataframe[:datalen,:]


        self.label_dict        = {imagename:label for imagename, label in zip(self.imagenames, np.array(images_dataframe))}


        self.perc_data   = perc_data
        self.rng         = np.random.RandomState(seed)
        self.n_files     = len(self.imagenames)

        means_CelebA   = np.array([0.5064, 0.4263, 0.3845])
        stds_CelebA    = np.array([0.1490, 0.1443, 0.1469])

        self.random_sample = random_sample

        self.transform = transforms.Compose([transforms.Pad((7,3,7,3)), transforms.ToTensor(), transforms.Normalize(mean=means_CelebA, std=stds_CelebA)])
        self.labels = None

        if self.preload_images:
            self.all_images = []    
            for imagename in tqdm(self.imagenames, desc='Loading Images into RAM...'):
                self.all_images.append(self.transform(Image.open(self.image_sourcepath+'/'+imagename)))

    def __getitem__(self, idx):
        self.rng  = np.random.RandomState(idx+self.rng.randint(0,1e8))
        idx_of_choice = self.rng.randint(0,len(self.imagenames))
        if self.preload_images:
            image    = self.all_images[idx_of_choice]
        else:
            image     = self.transform(Image.open(self.image_sourcepath+'/'+self.imagenames[idx_of_choice]))

        if self.labels is not None:
            return {'Input Image':image, 'Cluster Label':self.labels[idx_of_choice], 'Label Vector':self.label_dict[self.imagenames[idx_of_choice]]}
        else:
            return {'Input Image':image, 'Label Vector':self.label_dict[self.imagenames[idx_of_choice]]}

    def __len__(self):
        return self.n_files
