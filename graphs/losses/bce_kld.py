"""
An example for loss class definition, that will be used in the agent
"""
import torch.nn as nn
from torch.nn import functional as F


class BCE_KLDLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCE_KLDLoss, self).__init__()
        self.loss = nn.backends()

    # def forward(self, logits, labels):
    #     loss = self.loss(logits, labels)
    #     return loss
		# # Reconstruction + KL divergence losses summed over all elements and batch
		
    def forward(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
				# see Appendix B from VAE paper:
				# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
				# https://arxiv.org/abs/1312.6114
				# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
