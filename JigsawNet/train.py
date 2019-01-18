#Created on: Date
#@author: Karsten Roth
#based on implementation by Biagio Brattoli
"""============================================"""
"""============ LIBRARIES ====================="""
"""============================================"""
import os, sys, numpy as np, argparse, time, gc, datetime, pickle as pkl, random
# os.chdir('/home/karsten_dl/Dropbox/Projects/current_projects/manifoldlearning/Network_Training/JigsawNet')
from tqdm import tqdm, trange
import torch, torch.nn as nn
import network_libary as netlib
import auxiliaries as aux
import CelebA_dataset_Jigsaw as dataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

"""============================================"""
"""============ ARGUMENTS ====================="""
"""============================================"""
parser = argparse.ArgumentParser()
# parser.add_argument('--save_path', default='/export/home/kroth/Project_Manifold/SAVEDATA', type=str, help='Path to save training information')
# parser.add_argument('--data_path', default='/export/home/kroth/Project_Manifold/LOADDATA/IMAGES', type=str, help='Path to save training information')
# parser.add_argument('--perm_path', default='/export/home/kroth/Project_Manifold/manifoldlearning/Network_Training/JigsawNet/permutations_classes-200_tiles-9.npy', type=str, help='Path to save training information')
parser.add_argument('--save_path', default=os.getcwd()+'/Networks', type=str, help='Path to save training information')
parser.add_argument('--data_path', default=os.getcwd()+'/../../METRICLEARNING/Datasets', type=str, help='Path to save training information')
parser.add_argument('--perm_path', default=os.getcwd()+'JigsawNet/Permutations/permutations_classes-200_tiles-9.npy', type=str, help='Path to save training information')
parser.add_argument('--dataset',   default='cub200', type=str, help='Path to save training information')

parser.add_argument('--savename',        default='', type=str, help='Save Folder Name.')

parser.add_argument('--num_classes', default=200, type=int, help='Number of permutation to use.')
parser.add_argument('--num_tiles',   default=9, type=int, help='Number of tiles to use for JIGSAW puzzeling.')

parser.add_argument('--lr',          default=0.001, type=float, help='learning rate for SGD optimizer')
parser.add_argument('--n_epochs',    default=70, type=int, help='number of total epochs for training.')
parser.add_argument('--bs',          default=32, type=int, help='Batch Size.')
parser.add_argument('--step_size',   default=35, type=int, help='step size')
parser.add_argument('--gamma',       default=0.2, type=int, help='gamma')

parser.add_argument('--verbose_idx', default=200, type=int, help='Number of iterations to run before printing information.')
parser.add_argument('--gpu',         default=0, type=int, help='Choice of GPU.')
parser.add_argument('--kernels',     default=8, type=int, help='number of CPU core for loading')
parser.add_argument('--seed',        default=1, type=int, help='Seed for reproducability.')

parser.add_argument('--all_to_ram',  action='store_true', help='Preload data in dataloader to RAM')
opt = parser.parse_args([])


opt.save_path += '/'+opt.dataset
opt.data_path += '/'+opt.dataset




"""===================================================================="""
"""============ TRAINING AND VALIDATION FUNCTIONS ====================="""
"""===================================================================="""
############## TRAINER ###############
def train_one_epoch(opt, epoch, net, optimizer, criterion, dataloader, Metrics):
    start = time.time()

    epoch_coll_loss, epoch_coll_acc = [],[]

    data_iter = tqdm(dataloader, desc='Training...')
    for iter_idx, file_dict in enumerate(data_iter):

        prediction  = net(file_dict['Tiles'].type(torch.FloatTensor).to(opt.device))
        loss        = criterion(prediction, file_dict['Target'].type(torch.LongTensor).to(opt.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (np.argmax(prediction.cpu().detach().numpy(),axis=1)==target.cpu().detach().numpy()).reshape(-1)

        #--- Get Scores ---
        epoch_coll_acc.extend(list(acc))
        epoch_coll_loss.append(float(loss.item()))

        if iter_idx==len(dataloader)-1:
            data_iter.set_description('Epoch {0}: Loss [{1:.5f}] | Acc [{2:.3f}]'.format(epoch, np.mean(epoch_coll_loss), np.mean(epoch_coll_acc)))

    # Empty GPU cache
    torch.cuda.empty_cache()

    # Save Training Epoch Metrics
    Metrics['Train Loss'].append(np.round(np.mean(epoch_coll_loss),4))
    Metrics['Train Acc'].append(np.round(np.mean(epoch_coll_acc),4))
    Metrics['Train Time'].append(np.round(time.time()-start,4))



############## VALIDATOR ###############
def evaluate(opt, epoch, net, criterion, dataloader, Metrics):
    global best_val_acc
    start = time.time()

    epoch_coll_loss, epoch_coll_acc = [],[]

    data_iter = tqdm(dataloader, desc='Evaluating...')
    for iter_idx, file_dict in enumerate(data_iter):

        prediction  = net(file_dict['Tiles'].type(torch.FloatTensor).to(opt.device))
        loss      = criterion(prediction, file_dict['Target'].type(torch.LongTensor).to(opt.device))

        acc = (np.argmax(prediction.cpu().detach().numpy(), axis=1)==target.cpu().detach().numpy()).reshape(-1)

        #--- Get Scores ---
        epoch_coll_acc.extend(list(acc))
        epoch_coll_loss.append(float(loss.item()))

        if iter_idx==len(dataloader)-1:
            data_iter.set_description('Epoch {0}: Loss [{1:.5f}] | Acc [{2:.5f}]'.format(epoch, np.mean(epoch_coll_loss), np.mean(epoch_coll_acc)))

    # Empty GPU cache
    torch.cuda.empty_cache()

    if np.mean(epoch_coll_acc)>best_val_acc:
        set_checkpoint(model, epoch, opt, progress_saver)
        best_val_acc = np.mean(epoch_coll_acc)

    # Save Training Epoch Metrics
    Metrics['Val Loss'].append(np.round(np.mean(epoch_coll_loss),4))
    Metrics['Val Acc'].append(np.round(np.mean(epoch_coll_acc),4))
    Metrics['Val Time'].append(np.round(time.time()-start,4))


############## CHECKPOINT SETTER ###############
def set_checkpoint(model, epoch, opt, progress_saver):
    torch.save({'epoch': epoch+1, 'state_dict':model.state_dict(), 'opt':opt,
                'progress':progress_saver}, opt.savepath+'/checkpoint.pth.tar')



"""=================================================="""
"""============ ACTUAL TRAINING ====================="""
"""=================================================="""
def main():
    ############## SEEDS #########################################
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic=True

    bcolor = aux.bcolors


    ############## GPU SETTINGS ##################################
    os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]  = str(opt.gpu)


    ############## SET DATASETS AND -LOADER ######################
    print(bcolor.HEADER+'Setting DataLoader... '+bcolor.ENDC)
    if opt.dataset=='cars196':          train_dataset, val_dataset = aux.give_CARS196_datasets(opt)
    if opt.dataset=='cub200':           train_dataset, val_dataset = aux.give_CUB200_datasets(opt)
    if opt.dataset=='online_products':  train_dataset, val_dataset = aux.give_OnlineProducts_datasets(opt)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.bs, num_workers=opt.kernels, shuffle=True, pin_memory=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset,   batch_size=opt.bs, num_workers=opt.kernels, pin_memory=True)
    opt.num_classes    = len(train_dataset.avail_classes)
    print('Done.')


    ############## INITIALIZE NETWORK ######################
    print(bcolor.HEADER+'Setting up Network & Log-Files... '+bcolor.ENDC,end='')

    #################### CREATE SAVING FOLDER ###############
    date = datetime.datetime.now()
    time_string = '{}-{}-{}-{}-{}-{}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second)
    checkfolder = opt.save_path+'/{}_JigsawNetwork_'.format(opt.dataset)+time_string
    counter     = 1
    while os.path.exists(checkfolder):
        checkfolder = opt.save_path+'_'+str(counter)
        counter += 1
    os.makedirs(checkfolder)
    opt.save_path = checkfolder


    #################### SAVE OPTIONS TO TXT ################
    with open(opt.save_path+'/Parameter_Info.txt','w') as f:
        f.write(aux.gimme_save_string(opt))
    pkl.dump(opt,open(opt.save_path+"/hypa.pkl","wb"))


    #################### CREATE LOGGING FILES ###############
    InfoPlotter   = aux.InfoPlotter(opt.save_path+'/InfoPlot.svg')
    full_log      = aux.CSV_Writer(opt.save_path +'/log_epoch.csv', ['Epoch', 'Train Loss', 'Val Loss', 'Val Acc'])
    Progress_Saver= {'Train Loss':[], 'Val NMI':[], 'Val Recall Sum':[]}


    #################### SETUP JIGSAW NETWORK ###################
    opt.device = torch.device('cuda')
    model = netlib.NetworkSelect(opt)
    print('JIGSAW Setup for [{}] complete with #weights: {}'.format(opt.arch, aux.gimme_params(model)))
    _ = model.to(opt.device)

    global best_val_acc
    best_val_acc = 0


    ################### SET OPTIMIZATION SETUP ##################
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, opt.tau, gamma=opt.gamma)
    print('Done.')



    ############## START TRAINING ###########################
    print(bcolor.BOLD+bcolor.WARNING+'Starting Jigsaw Network Training!\n'+bcolor.ENDC+bcolor.ENDC)



    for epoch in range(opt.n_epochs):
        scheduler.step()

        ### Training ###
        train_one_epoch(opt, epoch, net, optimizer, criterion, train_dataloader, Metrics)

        ### Validating ###
        evaluate(opt, epoch, net, criterion, val_dataloader, Metrics)


        ###### Logging Epoch Data ######
        full_log.log([len(Metrics['Train Loss']), Metrics["Train Loss"][-1], Metrics["Train Acc"][-1], Metrics["Val Acc"][-1]])


        ###### Generating Summary Plots #######
        InfoPlotter.make_plot(range(epoch+1), Progress_Saver['Train Loss'], Progress_Saver['Val Acc'], ['Train Loss', 'Val Acc'])



if __name__ == "__main__":
    main()
