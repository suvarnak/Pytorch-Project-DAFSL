"""
Mnist Main agent, as mentioned in the tutorial
"""
import numpy as np

from tqdm import tqdm
import shutil
import os
import random
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

from agents.base import BaseAgent

from graphs.models.dafsl_concept_discriminator import DAFSL_ConceptDiscriminatorModel
from datasets.dafsl_discriminator_dataloader import DAFSLDiscriminatorDataLoader

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics

cudnn.benchmark = True


class DAFSL_ConceptClassifierAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info(
                "WARNING: You have a CUDA device, so you should probably enable CUDA")

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
        domain_name = self.config.current_domain
        img_root_folder = self.config.discriminator_datasets_root_dir
        self.class_labels = os.listdir(os.path.join(
            img_root_folder, domain_name, "train"))
        # define models
        no_class_labels = len(self.class_labels)
        self.model = DAFSL_ConceptDiscriminatorModel(
            config=self.config, no_class_labels=no_class_labels)

        # define optimizer
        self.optimizer = optim.SGD(self.model.parameters(
        ), lr=self.config.learning_rate, momentum=self.config.momentum)

        #Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = SummaryWriter(
            log_dir=self.config.summary_dir, comment='DAFSL_DCC')

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=False, domain_name='dummy'):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        dst = os.path.join("models", "discriminator", domain_name)
        print("creating.......", dst, os.path.exists(dst))
        if os.path.exists(dst) == False:
            print("creating.......", dst)
            os.makedirs(dst)
        torch.save(state, dst + 'model.pth.tar')
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(dst + 'model.pth.tar',
                            os.path.join(dst, 'model_best.pth.tar'))

    def load_checkpoint(self, filename="model.pth.tar", domain_name=""):
        filename = os.path.join(
            "models", "discriminator", domain_name+ filename)
        # define loss
        self.loss = nn.CrossEntropyLoss()

        try:
            self.logger.info(
                "Loading checkpoint '{}' for domain {}".format(filename, domain_name))
            checkpoint = torch.load(filename)
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(
                self.config.checkpoint_dir))
            self.logger.info("**First time to train**")
            self.current_epoch = 0
            self.current_iteration = 0
            self.best_valid_loss = 0 
    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate_acrossdomains()
            else:
                self.train_alldomains()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train_alldomains(self):
        """
        This function will the operator
        :return:
        """
        for domain_name in self.config.data_domains.split(','):
            self.train_discriminative_model(domain_name)

    def train_discriminative_model(self, domain_name):
        self.logger.info(
            "Training the discriminative model for {} domain".format(domain_name))
        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file,
                             domain_name=domain_name)
        self.data_loader = DAFSLDiscriminatorDataLoader(
            config=self.config, domain_name=domain_name)
        try:
            if self.config.mode == 'test':
                self.validate(domain_name=domain_name)
            else:
                self.train(domain_name=domain_name)
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def validate_acrossdomains(self):
        """
        This function will the operator
        :return:
        """
        pass

    def train(self, domain_name):
        """
        Main training function, with per-epoch model saving
        """
        self.criterion = nn.CrossEntropyLoss()  # BCE_KLDLoss(self.model)
        summary(self.model, input_size=(3, self.config.image_size, self.config.image_size))
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch(domain_name=domain_name)
            valid_loss = self.validate(domain_name=domain_name)
            is_best = valid_loss > self.best_valid_loss
            if is_best:
                self.best_valid_loss = valid_loss
            self.logger.info("Saving model checkpoint for epoch")
            self.save_checkpoint(is_best=is_best, domain_name=domain_name)

    def train_one_epoch(self, domain_name):
        """
        One epoch of trainings
        :return:
        """

        self.model.train()
        epoch_lossD = AverageMeter()
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            # self.logger.info(str(batch_idx))
            # self.logger.info(str(target.size()))
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            #self.logger.info(str(output.size()))
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            epoch_lossD.update(loss.item())
            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx *
                    len(data), len(self.data_loader.train_loader.dataset),
                    100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
            self.current_iteration += 1
            self.summary_writer.add_scalar(
                "epoch/Discriminator_loss", epoch_lossD.val, self.current_iteration)
        #self.visualize_one_epoch()
        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " +
                         " - Discriminator Loss-: " + str(epoch_lossD.val) + "for domain "+domain_name)

    def validate(self, domain_name):
        """
        One cycle of model validation
        :return:
        """
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for data in self.data_loader.test_loader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1,keepdim=True)
                test_loss += F.cross_entropy(outputs, labels, size_average=False).item()  # sum up batch loss
                pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()
        test_loss /= len(self.data_loader.test_loader.dataset)
        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.data_loader.test_loader.dataset),
            100. * correct / len(self.data_loader.test_loader.dataset)))
        return test_loss

	
    def visualize_one_epoch(self):
        """
        One epoch of visualizing
        :return:
        """
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.data_loader.test_loader):
                testimgs, _ = data  # data.to(self.device)
                output = self.model(testimgs)
                pred = output.max(1, keepdim=True)[1]
                #print(list(pred.size()), pred)
                plt.figure()
                img = testimgs[batch_idx]
                #print(list(img.size()))
                img = img.permute(1, 2, 0)
                # print(list(img.size()))
                # self.data_loader.plot_samples_per_epoch(img,self.current_epoch)
                plt.imshow(img.numpy())
                plt.xlabel("Predicted Class"+str(pred[batch_idx].item()))
                #plt.imsave("PredictedClass"+str(pred[batch_idx].item()),img)
        # plt.show()

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        self.logger.info(
            "Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.close()
        self.data_loader.finalize()
