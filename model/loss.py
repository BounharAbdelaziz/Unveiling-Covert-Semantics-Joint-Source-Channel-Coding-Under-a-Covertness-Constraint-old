import torch
import torch.nn as nn
from scipy.stats import entropy
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

class ReconstructionLoss(nn.Module):
    def __init__(self, loss_fct='L2'):
        super(ReconstructionLoss, self).__init__()

        # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if loss_fct.upper() == 'L2':
            self.loss = nn.MSELoss().to(device)
        elif loss_fct.upper() == 'L1':
            self.loss = nn.L1Loss().to(device)
        else:
            raise ValueError(f"Invalid loss function: {loss_fct}. Choose from ['L1', 'L2'].")
        self.eps = 1e-10

    def forward(self, x, y):
        loss = self.eps + self.loss(x, y)
        return loss
    
# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()

        # training device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # the loss function
        self.loss = nn.CrossEntropyLoss().to(device)
        self.eps = 1e-10

    def forward(self, logits, labels):
        loss = self.eps + self.loss(logits, labels)
        return loss
    
# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

class PowerConstraintLoss(nn.Module):
    def __init__(self, maximum_power=1.0, exponent=2.0):
        super(PowerConstraintLoss, self,).__init__()

        # maximum power constraint
        self.maximum_power = maximum_power
        # dictates how much we care about the power constraint
        self.exponent = exponent

    def forward(self, x):
        # Compute the average power
        avg_power = torch.mean(torch.sum(x**2, dim=1))
        # Compute the error term
        error = (avg_power/self.maximum_power)**self.exponent
        # Compute the loss
        loss = torch.abs(1 - error)
        
        return loss
    
# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, kappa):
        super(SoftHistogram, self).__init__()
        
        # training device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # parameters for the histogram
        self.bins = bins
        self.min = min
        self.max = max
        self.kappa = kappa
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5).to(self.device)

    def forward(self, x):
        s = 0
        for i in range(len(x)):
            z = torch.unsqueeze(x[i], 0) - torch.unsqueeze(self.centers, 1)
            z = torch.sigmoid(self.kappa * (z + self.delta/2)) - torch.sigmoid(self.kappa * (z - self.delta/2))
            z = z.sum(dim=1)
            s+= z
            
        return s

# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #
    
class CovertnessConstraintLoss(nn.Module):
    def __init__(self, bins=500, min_val=-15, max_val=15, kappa=1, noise_power=1.0, delta_n=0.1):
        super(CovertnessConstraintLoss, self).__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        self.kappa = kappa
        self.noise_power = noise_power
        self.delta_n = delta_n
        self.softhist = SoftHistogram(bins=bins, min=min_val, max=max_val, kappa=kappa)


    def forward(self, z_1, z_0):
        
        batch_size = z_1.size(0)

        # # Compute differentiable histogram for z_1
        # hist_1 = self.softhist(z_1)

        # # Compute Gaussian distribution for z_0 (it's a known distribution, the distribution of the noise that is supposed gaussian, so we don't need to compute a differentiable histogram)
        # z_0_bins = torch.linspace(self.min_val, self.max_val, self.bins)
        # gaussian_hist_0 = torch.exp(-0.5 * ((z_0_bins / self.noise_power) ** 2)) / (self.noise_power * (2 * torch.pi) ** 0.5)
        
        
        # batch_size = z_1.size(0)

        # kl_divs = []
        # for i in range(batch_size):
        #     # Compute differentiable histogram for z_1[i]
        #     hist_1 = self.softhist(z_1[i])

        #     # Compute Gaussian distribution for z_0
        #     z_0_bins = torch.linspace(self.min_val, self.max_val, self.bins)
        #     gaussian_hist_0 = torch.exp(-0.5 * ((z_0_bins / self.noise_power) ** 2)) / (self.noise_power * (2 * torch.pi) ** 0.5)

        #     # Compute KL divergence
        #     kl_div = torch.sum(hist_1 * (torch.log(hist_1) - torch.log(gaussian_hist_0)))
        #     kl_div = torch.max(torch.tensor(0.0), kl_div)
        #     kl_divs.append(kl_div)

        # # Convert list to tensor
        # kl_divs = torch.stack(kl_divs)

        # print(f'kl_divs: {kl_divs}')

        # # Plot histograms
        # plt.figure(figsize=(10, 5))
        # plt.plot(gaussian_hist_0.detach().numpy(), label='Gaussian distribution for z_0')
        # plt.plot(hist_1.detach().numpy(), label='Histogram for z_1')
        # plt.title('Comparison with Gaussian Distribution')
        # plt.xlabel('Bin')
        # plt.ylabel('Frequency')
        # plt.legend()
        # plt.show()

        # plt.savefig(f'plots/training/histogram_covert.png')

        # # Compute KL divergence
        # kl_div = torch.sum(hist_1 * (torch.log(hist_1) - torch.log(gaussian_hist_0)))
        # kl_div = torch.max(torch.tensor(0.0), kl_div)

        # print(f'kl_div: {kl_div}')
        
        # Compute differentiable histograms
        hist_0 = self.softhist(z_0)
        hist_1 = self.softhist(z_1)

        # Compute KL divergence
        kl_div = max(0, torch.sum(hist_1 * (torch.log(hist_1) - torch.log(hist_0))))
        
        kl_div = torch.abs(torch.sum(hist_1 * (torch.log(hist_1) - torch.log(hist_0))))
        loss = torch.abs(kl_div - self.delta_n * batch_size)

        # print(f'kl_div: {kl_div}')
        # print(f'loss: {loss}')
        # print('-----------------------------')

        # # Plot histograms
        # plt.figure(figsize=(10, 5))
        # plt.plot(hist_0.cpu().detach().numpy(), label='Distribution under $\mathcal{H}=0$')
        # plt.plot(hist_1.cpu().detach().numpy(), label='Distribution under $\mathcal{H}=1$')
        # plt.title('Output distribution under the two hypotheses covert model')
        # plt.xlabel('Bin')
        # plt.ylabel('Frequency')
        # plt.legend()
        # plt.show()
        # plt.savefig(f'plots/training/histogram_covert.png')
        # plt.close()

        # return loss
        return kl_div
    
# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #
  