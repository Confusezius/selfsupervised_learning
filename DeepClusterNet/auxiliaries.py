import matplotlib.pyplot as plt
import os, numpy as np, csv, time
import torch.nn as nn
import torch
from tqdm import tqdm, trange

################## DATA LOGGER #####################
class DataLogger():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0,0,0,0
        self.log = []

    def update(self, val, n=1):
        self.val   = val
        self.sum   = self.sum + val * n
        self.count = self.count + n
        self.avg   = self.sum/self.count
        self.log.append(val)


################## SET OF DATA LOGGERS #####################
class LogSet():
    def __init__(self, lognames):
        self.lognames = lognames
        self.logs     = {logname:DataLogger() for logname in lognames}


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

################## PLOT SUMMARY IMAGE #####################
class InfoPlotter():
    def __init__(self, save_path, title='Training Log', figsize=(15,10)):
        self.save_path = save_path
        self.title     = title
        self.figsize   = figsize

    def make_plot(self, x, y):
        plt.style.use('ggplot')
        f,ax = plt.subplots(1)
        ax.plot(x, y)
        ax.set_title(self.title)
        f.set_size_inches(self.figsize[0], self.figsize[1])
        f.savefig(self.save_path)
        plt.close()



################ Clustering Utilities #################
def prepare_model_4_clustering(model):
    #### Remove last fully-connected and activation layer
    model.top_layer = None
    model.MLP_wo_top = nn.Sequential(*list(model.MLP_wo_top.children())[:-1])
    # return model

def rebuild_model_4_training(model, num_classes):
    #### Set last fully_connected layer again (or use fully-convolutional with GAP)
    mlp = list(model.MLP_wo_top.children())
    mlp.append(nn.ReLU(inplace=True).cuda())
    model.MLP_wo_top = nn.Sequential(*mlp)
    model.top_layer  = nn.Linear(model.final_units, num_classes)
    model.top_layer.weight.data.normal_(0,0.01)
    model.top_layer.bias.data.zero_()
    model.top_layer.cuda()
    # return model


################# SAVE NETWORK AND TRAIN FOR ONE EPOCH ###########################
def set_checkpoint(model, epoch, optimizer1, optimizer2, opt):
    torch.save({'epoch':epoch+1,
                'arch':opt.arch,
                'state_dict':model.state_dict(),
                'features optim state dict':optimizer1.state_dict(),
                'top optim state dict':optimizer2.state_dict()
                }, opt.savefolder+'/checkpoint.tar.pth')



def train_network_one_epoch(image_dataloader, model, loss_function, optimizer, opt, epoch, logs, csv_logger_iter, csv_logger_epoch):
    _ = model.train()

    optimizer_last_layer = torch.optim.SGD(model.top_layer.parameters(), lr=opt.lr, weight_decay=opt.l2)

    iterator = tqdm(image_dataloader, position=1)
    iterator.set_description('Network Training with Loss: ----- ')

    loss_collect_epoch = []
    for i, file_dict in enumerate(iterator):
        end = time.time()

        target             = file_dict['Cluster Label'].to(opt.device)
        ###NOTE: This was set to .cuda(async=True) before
        input_image_tensor = file_dict['Input Image'].to(opt.device)

        prediction = model(input_image_tensor)
        loss       = loss_function(prediction, target)

        optimizer.zero_grad()
        optimizer_last_layer.zero_grad()

        loss.backward()

        optimizer.step()
        optimizer_last_layer.step()

        loss_collect_epoch.append(loss.item())

        if i%opt.iter_update==0 and i!=0:
            logs.logs['train time per iter'].update(np.mean(loss_collect_epoch), opt.iter_update)
            iterator.set_description('Loss: {}'.format(np.round(np.mean(loss_collect_epoch),6)))
            csv_logger_iter.log([i, np.mean(loss_collect_epoch), end-time.time()])

    logs.logs['loss per epoch'].update(np.mean(loss_collect_epoch))
    csv_logger_epoch.log([epoch, np.mean(loss_collect_epoch), end-time.time()])

    set_checkpoint(model, epoch, optimizer, optimizer_last_layer, opt)
