import warnings
warnings.filterwarnings("ignore")

import os, pickle, sys, time, argparse, imp, datetime, random
import faiss, numpy as np

from sklearn.metrics.cluster import normalized_mutual_info_score
from tqdm import tqdm, trange


import torch, faiss
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

os.chdir('/home/karsten_dl/Dropbox/Projects/current_projects/manifoldlearning/Network_Training/DeepClusterNet')

import clustering, auxiliaries as aux, baseline_conv as netlib
import CelebA_dataset_DeepCluster as dataset
import pandas as pd


def main():
    ################### INPUT ARGUMENTS ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cluster',  default=200, type=int, help='number of kmeans cluster')
    parser.add_argument('--lr',           default=0.05, type=float, help='Learning Rate')
    parser.add_argument('--arch',         default='alexnet', type=str, help='Learning Rate')
    parser.add_argument('--l2',           default=0.00001, type=float, help='L2-Weight-regularization')
    parser.add_argument('--perc_data',    default=1, type=float, help='L2-Weight-regularization')
    parser.add_argument('--cluster_intv', default=1, type=int, help='Number of epchs before recomputing supervised cluster labels')
    parser.add_argument('--kernels',      default=8, type=int, help='Number of cores to use.')
    parser.add_argument('--bs',           default=2, type=int, help='Mini-Batchsize to use.')
    parser.add_argument('--seed',         default=1, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--n_epochs',     default=100, type=int, help='Number of training epochs.')
    parser.add_argument('--gpu',     default=0,   type=int, help='GPU to use.')
    parser.add_argument('--iter_update',  default=200, type=int, help='Number of iterations before each log entry.')
    # parser.add_argument('--save_path',    default='/export/home/kroth/Project_Manifold/SAVEDATA', type=str, help='Path to save training information')
    # parser.add_argument('--data_path',    default='/export/home/kroth/Project_Manifold/LOADDATA', type=str, help='Path to load training information')
    parser.add_argument('--save_path',    default='/home/karsten_dl/Dropbox/Data_Dump/DeepClustering/SAVEDATA/DeepClusterNetwork_Training', type=str, help='Path to save training information')
    parser.add_argument('--data_path',    default='/home/karsten_dl/Dropbox/Data_Dump/DeepClustering/LOADDATA', type=str, help='Path to save training information')
    parser.add_argument('--savename',     default='', type=str, help='Save Folder Name')
    parser.add_argument('--pca_dim',      default=128, type=int, help='Use sobel filter as initialization.')
    parser.add_argument('--no_sobel',          action='store_true', help='Use sobel filter as initialization.')
    parser.add_argument('--no_dim_reduction',  action='store_true', help='Use sobel filter as initialization.')
    parser.add_argument('--make_umap_plots',   action='store_true', help='Use sobel filter as initialization.')
    parser.add_argument('--dont_preload_images',    action='store_true', help='Use sobel filter as initialization.')
    opt = parser.parse_args(['--dont_preload_images', '--perc_data', '0.01'])

    opt.use_sobel         = not opt.no_sobel
    opt.use_dim_reduction = not opt.no_dim_reduction
    opt.preload_images    = not opt.dont_preload_images



    ################ FIX SEEDS ##################
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    rng = np.random.RandomState(opt.seed)


    os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]  = str(opt.gpu)


    opt.device = torch.device('cuda')


    ################ SET DATALOADER ###########
    print('\033[92m'+'Setting up DataLoader... '+'\033[0m')
    image_dataset    = dataset.CelebA_Dataset(opt.data_path, perc_data=opt.perc_data, preload_images=opt.preload_images)
    image_dataloader = DataLoader(image_dataset, batch_size=opt.bs, num_workers=opt.kernels, pin_memory=True)
    input_size = image_dataset[0]['Input Image'].numpy().shape[1:]
    print('Done.\n')


    ################ LOAD MODEL ################
    print('\033[92m'+'Setting up network [{}]... '.format(opt.arch.upper())+'\033[0m')
    imp.reload(netlib)
    model = netlib.AlexNet(opt.num_cluster, input_size, use_sobel = opt.use_sobel, use_batchnorm=True)
    _ = model.to(device)


    ################ SET OPTIMIZER ############
    optimizer     = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr = opt.lr, momentum=0.9, weight_decay=opt.l2)
    loss_function = nn.CrossEntropyLoss().cuda()
    print('Done with [#weights] {}.\n'.format(aux.gimme_params(model)))


    ############### SET SAVEFOLDER ############
    savename = model.name
    if opt.savename == '':
        savename += '___'+opt.savename
    date = datetime.datetime.now()
    savename += '___'+'-'.join(str(x) for x in [date.year, date.month, date.day, date.hour, date.minute, date.second])
    opt.savename = savename
    opt.savefolder = opt.save_path + '/' + opt.savename
    checkfolder, counter   = opt.savefolder, 1

    while os.path.exists(checkfolder):
        checkfolder = opt.savefolder+ '_' +str(counter)
        counter += 1
    opt.savefolder = checkfolder
    os.makedirs(opt.savefolder)



    ################ SET LOGGING ##############
    plot_generator = aux.InfoPlotter(opt.savefolder + '/training_progress.png')
    logs           = aux.LogSet(['cluster time', 'train time per iter', 'train time per epoch', 'loss per iter', 'loss per epoch'])
    CSV_Log_per_batch   = aux.CSV_Writer(opt.savefolder+'/training_log_iter-'+str(opt.iter_update)+'.csv', columns=['Iteration','Loss','Elapsed Time'])
    CSV_Log_per_epoch   = aux.CSV_Writer(opt.savefolder+'/training_log_epoch-'+str(opt.n_epochs)+'.csv', columns=['Epoch', 'Loss', 'Elapsed Time'])




    ################ Training Function ###############
    print('\033[93m'+'Starting Training...\n'+'\033[0m')
    epoch_iterator = trange(opt.n_epochs, position=0)
    epoch_iterator.set_description('Running Training...')

    for epoch in epoch_iterator:
        opt.epoch = epoch
        # opt.make_umap_plots = True if epoch%3==0 else False

        image_dataloader.dataset.labels = None
        pseudolabels = compute_clusters_and_set_dataloader_labels(image_dataloader, model, opt)

        clustering.adjust_model_compute_clusters_and_set_dataloader_labels(image_dataloader, model, opt, logs)
        aux.train_network_one_epoch(image_dataloader, model, loss_function, optimizer, opt, epoch, logs, CSV_Log_per_batch, CSV_Log_per_epoch)
        epoch_iterator.set_description('Network Training | Curr Loss: {} | Best Loss: {}'.format(logs.logs['loss per epoch'].log[-1], np.min(logs.logs['loss per epoch'].log)))
        if epoch>=2:
            plot_generator.make_plot(range(epoch+1), logs.logs['loss per epoch'].log)





################################################################################################
if __name__=='__main__':
    main()
