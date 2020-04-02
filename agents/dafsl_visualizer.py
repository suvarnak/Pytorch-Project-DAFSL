"""
Domain Agnostic FSL  Visualizer agent 
"""
import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim import Adam
from agents.base import BaseAgent

from graphs.models.dafsl_cae_model import DAFSL_CAEModel
from datasets.dafsl_data_loader import DAFSLDataLoader
from graphs.losses.bce_kld import BCE_KLDLoss

from torchviz import make_dot
from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics
import matplotlib.pyplot as plt
from torchsummary import summary

cudnn.benchmark = True


class DAFSLVisualizerAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        print(torch.__version__)
        # define models
        self.model = DAFSL_CAEModel()

        # define data_loader
        self.data_loader = DAFSLDataLoader(config=config)


        self.current_epoch = 0
        self.current_iteration = 0

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)


    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")



    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.visualize()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def visualize(self):
        """
        Main training loop
        :return:
        """
								# define the optimization
        summary(self.model, input_size=(3, 224, 224))
        self.criterion = MSELoss()#BCE_KLDLoss(self.model)
        for epoch in range(1, self.config.max_epoch + 1):
            self.visualize_one_epoch()
            self.current_epoch += 1

    def visualize_one_epoch(self):
        """
        One epoch of visualizing
        :return:
        """
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, data  in enumerate(self.data_loader.test_loader):
                testimgs, _ = data  #data.to(self.device)
                generated_testimgs = self.model(testimgs)
                #generated_testimgs = generated_testimgs[0]  				
								#make_dot(generated_img[0])
                print(list(generated_testimgs.size()))
                #print(list(testimgs.size()))
                #plt.figure()
                #img = testimgs[batch_idx]
                img = generated_testimgs #.reshape((generated_testimgs.size()[0], 3,224,224))
                #print(list(img.size()))
                #img  = img.permute(0,3,1,2)
                #print(list(img.size()))
                self.data_loader.plot_samples_per_epoch(img,self.current_epoch)
								#plt.imshow(img.numpy())



    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
