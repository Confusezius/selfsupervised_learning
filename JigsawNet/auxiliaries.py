import argparse, itertools as it, numpy as np, os, csv, matplotlib.pyplot as plt
from tqdm import trange, tqdm
import scipy.spatial.distance as ssd






################# FUNCTIONS TO RETURN TRAIN/VAL PYTORCH DATASETS FOR CUB200, CARS196 AND STANFORD ONLINE PRODUCTS ####################################
def give_CUB200_datasets(opt):
    image_sourcepath  = opt.source_path+'/images'
    image_classes = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x], key=lambda x: int(x.split('.')[0]))
    conversion    = {int(x.split('.')[0]):x.split('.')[-1] for x in image_classes}
    image_list    = {int(key.split('.')[0]):sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key) if '._' not in x]) for key in image_classes}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    image_dict    = {}
    for key, img_path in image_list:
        key = key-1
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    # seed = 1
    # random.seed(seed)

    keys = sorted(list(image_dict.keys()))
    # random.shuffle(keys)
    train,test = keys[:len(keys)//2], keys[len(keys)//2:]

    train_image_dict, val_image_dict = {key:image_dict[key] for key in train},{key:image_dict[key] for key in test}

    train_split = BaseTripletDataset(train_image_dict, opt, anchor_only=opt.loss=='nca')
    val_split   = BaseTripletDataset(val_image_dict,   opt, anchor_only=True)
    train_split.conversion = conversion
    val_split.conversion   = conversion
    return train_split, val_split


def give_CARS196_datasets(opt):
    image_sourcepath  = opt.source_path+'/images'
    image_classes = sorted([x for x in os.listdir(image_sourcepath)])
    conversion    = {i:x for i,x in enumerate(image_classes)}
    image_list    = {i:sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key)]) for i,key in enumerate(image_classes)}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    image_dict    = {}
    for key, img_path in image_list:
        key = key-1
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    seed = 1
    random.seed(seed)

    keys = list(image_dict.keys())
    random.shuffle(keys)
    train,test = keys[:len(keys)//2], keys[len(keys)//2:]

    train_image_dict, val_image_dict = {key:image_dict[key] for key in train},{key:image_dict[key] for key in test}

    train_split = BaseTripletDataset(train_image_dict, opt, anchor_only=opt.loss=='nca')
    val_split   = BaseTripletDataset(val_image_dict,   opt, anchor_only=True)
    train_split.conversion = conversion
    val_split.conversion   = conversion
    return train_split, val_split


def give_OnlineProducts_datasets(opt):
    image_sourcepath  = opt.source_path+'/images'
    training_files = pd.read_table(opt.source_path+'/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
    test_files     = pd.read_table(opt.source_path+'/Info_Files/Ebay_test.txt', header=0, delimiter=' ')

    conversion = {}
    for class_id, super_class_id, path in zip(training_files['class_id'],training_files['super_class_id'],training_files['path']):
        conversion[class_id] = path.split('/')[0]
    for class_id, super_class_id, path in zip(test_files['class_id'],test_files['super_class_id'],test_files['path']):
        conversion[class_id] = path.split('/')[0]

    train_image_dict, val_image_dict  = {},{}
    for key, img_path in zip(training_files['class_id'],training_files['path']):
        key = key-1
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(image_sourcepath+'/'+img_path)

    for key, img_path in zip(test_files['class_id'],test_files['path']):
        key = key-1
        if not key in val_image_dict.keys():
            val_image_dict[key] = []
        val_image_dict[key].append(image_sourcepath+'/'+img_path)


    train_split = BaseTripletDataset(train_image_dict, opt, anchor_only=opt.loss=='nca')
    val_split   = BaseTripletDataset(val_image_dict,   opt, anchor_only=True)
    train_split.conversion = conversion
    val_split.conversion   = conversion
    return train_split, val_split



################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseTileDataset(Dataset):
    def __init__(self, image_dict, opt, anchor_only=False):
        self.n_files     = np.sum([len(image_dict[key]) for key in image_dict.keys()])

        self.pars        = opt
        self.image_dict  = image_dict
        self.anchor_only = anchor_only

        self.permutations   = np.load(perm_path)
        self.permutations   = self.permutations-self.permutations.min()

        self.avail_classes = sorted(list(self.image_dict.keys()))
        self.image_dict    = {i:self.image_dict[key] for i,key in enumerate(self.avail_classes)}

        transf_list = [transforms.RandomHorizontalFlip(0.5)]
        if opt.dataset=='cub200':
            means = np.array([0.47819992, 0.49399305, 0.4262326 ])
            stds  = np.array([0.05760238, 0.05675151, 0.06677961])
            transf_list.append(transforms.CenterCrop([256,256]))
        if opt.dataset=='cars196':
            means = np.array([0.4706145 , 0.46000465, 0.45479808])
            stds  = np.array([0.04725483, 0.04907224, 0.04912915])
            transf_list.extend([transforms.Resize((700,480)), transforms.CenterCrop([256,256])])
        if opt.dataset=='online_products':
            means   = np.array([0.57989254, 0.53863949, 0.50323734])
            stds    = np.array([0.07214612, 0.07111237, 0.07302282])
            transf_list.extend([transforms.Resize((400,400)), transforms.CenterCrop([256,256])])

        transf_list.extend([transforms.ToTensor(), transforms.Normalize(mean=means, std=stds)])
        self.transform = transforms.Compose(transf_list)


        ##### IF REQUIRED: PRE-TRANSFORM AND LOAD ALL IMAGES TO RAM ##################
        if self.pars.all_to_ram and not self.pars.feature_extraction_only:
            totensor               = transforms.ToTensor()
            self.toPIL             = transforms.ToPILImage()
            self.loaded_image_dict = {}
            for key,items in tqdm(self.image_dict.items(), desc='Preloading all images...'):
                self.loaded_image_dict[key] = [totensor(Image.open(item)) for item in items]



        if self.pars.feature_extraction_only:
            self.image_list = [[(x,key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
            self.image_list = [x for y in self.image_list for x in y]



    def ensure_rgb(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img

    def __getitem__(self, idx):
        if self.pars.all_to_ram:
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





################### PRINT COLOR CLASS ############################
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

################## COMPUTE TILE PERMUTATIONS WITH MAX HAMMING DISTANCE ###########################
def compute_permutations(num_tiles=9, num_classes=200, savepath=os.getcwd(), seed=1):
    # savepath = '/media/karsten_dl/QS/Data/Dropbox/Projects/Current_projects/manifoldlearning/SELFSUPERVISED/JigsawNet'
    # num_tiles, num_classes, seed = 9,100,1
    rng = np.random.RandomState(seed)

    avail_permutations    = np.array(list(it.permutations(list(range(num_tiles)), num_tiles)))
    n_permutations  = avail_permutations.shape[0]

    for i in trange(num_classes):
        if i==0:
            perm_idx       = rng.randint(n_permutations)
            perm_of_choice = np.array(avail_permutations[perm_idx:perm_idx+1])
        else:
            perm_of_choice = np.concatenate([perm_of_choice,avail_permutations[perm_idx:perm_idx+1]],axis=0)

        avail_permutations = np.delete(avail_permutations,perm_idx,axis=0)

        dist_to_existing_perms = ssd.cdist(perm_of_choice, avail_permutations, metric='hamming').mean(axis=0).flatten()
        perm_idx               = dist_to_existing_perms.argmax()

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    np.save(savepath+'/permutations_classes-{}_tiles-{}.npy'.format(num_classes, num_tiles),perm_of_choice)



################## WRITE TO CSV FILE #####################
class CSV_Writer():
    def __init__(self, save_path, columns):
        self.save_path = save_path
        self.columns   = columns

        with open(self.save_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(self.columns)

    def log(self, inputs):
        with open(self.save_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(inputs)


################# ACQUIRE NUMBER OF WEIGHTS #################
def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


################# GET SUMMARY SAVE STRING #################
def gimme_save_string(opt):
    varx     = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key],dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n\n'
    return base_str


################## PLOT SUMMARY IMAGE #####################
class InfoPlotter():
    def __init__(self, save_path, title='Training Log', figsize=(15,10)):
        self.save_path = save_path
        self.title     = title
        self.figsize   = figsize

    def make_plot(self, x, y1, y2, labels=['Training', 'Validation']):
        plt.style.use('ggplot')
        f,ax = plt.subplots(1)
        ax.set_title(self.title)
        ax.plot(x, y1, '-k', label=labels[0])
        axx = ax.twinx()
        axx.plot(x, y2, '-r', label=labels[1])
        f.set_size_inches(self.figsize[0], self.figsize[1])
        f.savefig(self.save_path)
        plt.close()
