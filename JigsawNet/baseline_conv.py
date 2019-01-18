##### LIBRARIES #####
import torch, torch.nn as nn, sys
import numpy as np, math




##### NETWORK #####
class BaseConv_Scaffold(nn.Module):
    def __init__(self, num_classes, funnel_type='fully-convolutional'):
        super(BaseConv_Scaffold, self).__init__()

        self.features = nn.Sequential(nn.Conv2d(3,32,5,1,2),nn.LeakyReLU(0.2), nn.Dropout2d(0.2),
                                      nn.MaxPool2d(2,2),
                                      nn.Conv2d(32,64,5,1,2),nn.LeakyReLU(0.2),nn.Dropout2d(0.2),
                                      nn.MaxPool2d(2,2),
                                      nn.Conv2d(64,92,5,1,2),nn.LeakyReLU(0.2),nn.Dropout2d(0.2),
                                      nn.MaxPool2d(2,2))

        self.funnel_type = funnel_type
        if funnel_type=='fully-connected':
            # nn.Linear(64 * 54 * 44, 256)
            self.funnel   = nn.Sequential(nn.Linear(92 * 27 * 22, 256),nn.LeakyReLU(0.2),
                                          nn.Linear(256, 256),nn.LeakyReLU(0.2),
                                          nn.Linear(256, 2))
        else:
            self.funnel   = nn.Sequential(nn.Conv2d(92,128,3,1,1),nn.LeakyReLU(0.2),nn.Dropout2d(0.2),
                                          nn.MaxPool2d(2,2),
                                          nn.Conv2d(128,128,3,1,1),nn.LeakyReLU(0.2),nn.Dropout2d(0.2),
                                          nn.Conv2d(128,64,3,1,1))

        self.output_layer =  nn.Sequential(nn.Linear(9*64, 512), nn.LeakyReLU(0.2), nn.Dropout(p=0.4),
                                           nn.Linear(512,num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(9):
            z = self.funnel(self.features(x[i]))
            z = torch.nn.functional.avg_pool2d(z, kernel_size=z.size()[2:]).view(z.size()[0],-1)
            x_list.append(z)

        x = torch.cat(x_list, dim=1)

        x = self.output_layer(x)

        return x



##### ALEXNET #####
class AlexNet(nn.Module):
    def __init__(self, num_classes, num_tiles, input_size, use_sobel=False, use_batchnorm=False, config=None):
        self.pars = {key:item for key,item in locals().items() if key not in ['self', '__class__']}

        super(AlexNet, self).__init__()

        self.pars['config']   = [(96, 11, 2, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M'] if self.pars['config'] is None else self.pars['config']
        # self.pars['config']   = [(96, 11, 2, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M'] if self.pars['config'] is None else self.pars['config']
        if self.pars['use_sobel']: self.__GiveSobel()
        self.__GiveConvFeatures()
        self.__GiveMLP()

        self.__initialize_weights()

    def __GiveConvFeatures(self):
        self.final_feature_map_size = np.array(self.pars['input_size'])
        layers, in_channels = [], 3 + 2*int(self.pars['use_sobel'])
        for c in self.pars['config']:
            if c == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
                self.final_feature_map_size = (np.floor((self.final_feature_map_size-3+2*0)//2)+1).astype(int)
            elif c == 'ML':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2), nn.LRN(local_size=5, alpha=0.0001, beta=0.75)]
                self.final_feature_map_size = (np.floor((self.final_feature_map_size-3+2*0)//2)+1).astype(int)
            else:
                conv2d = nn.Conv2d(in_channels, c[0], kernel_size=c[1], stride=c[2], padding=c[3])
                self.feature_out_channels = c[0]
                if self.pars['use_batchnorm']:
                    layers += [conv2d, nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = c[0]
                self.final_feature_map_size = (np.floor((self.final_feature_map_size-c[1]+2*c[3])//c[2])+1).astype(int)

        self.features = nn.Sequential(*layers)

    def __GiveSobel(self):
        # Initialize a conv-layer with sobel weigths
        grayscale = nn.Conv2d(3,1,kernel_size=1,stride=1,padding=0)
        grayscale.weight.data.fill_(1.0/3.0)
        grayscale.bias.data.zero_()
        sobel_filter = nn.Conv2d(1,2,kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0,0].copy_(torch.FloatTensor([[1,0,-1], [2,0,-2],[1,0,-1]]))
        sobel_filter.weight.data[1,0].copy_(torch.FloatTensor([[1,2,1],[0,0,0],[-1,-2,-1]]))
        sobel_filter.bias.data.zero_()

        self.sobel = nn.Sequential(grayscale, sobel_filter)
        for p in self.sobel.parameters():
            p.requires_grad = False

    def __GiveMLP(self):
        self.extend_length = int(self.feature_out_channels*np.prod(self.final_feature_map_size))
        self.MLP_before_merge = nn.Sequential(nn.Linear(self.extend_length, 1024),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(0.5))

        self.MLP_after_merge  = nn.Sequential(nn.Linear(self.pars['num_tiles']*1024, 4096),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(0.5))

        self.top_layer = nn.Linear(4096, self.pars['num_classes'])

    def __initialize_weights(self):
        for idx,module in enumerate(self.modules()):
            if isinstance(module, nn.Conv2d):
                n = np.prod(module.kernel_size[:2])*module.out_channels
                for i in range(module.out_channels):
                    module.weight.data[i].normal_(0, math.sqrt(2./n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0,0.01)
                module.bias.data.zero_()


    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(self.pars['num_tiles']):
            z = torch.cat([x[i],self.sobel(x[i])], dim=1) if self.pars['use_sobel'] else x[i]
            z = self.features(z)
            z = self.MLP_before_merge(z.view(B,-1))
            x_list.append(z.view([B,1,-1]))

        x = torch.cat(x_list, dim=1)
        x = self.MLP_after_merge(x.view(B,-1))
        x = self.top_layer(x)

        return x
