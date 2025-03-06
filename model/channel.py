import torch
import numpy as np


class Channel():
    
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
    
    def __init__(self, 
                    type='AWGN', 
                    SNR=10
                ):
        
        super(Channel, self).__init__()
        # the type of the channel
        self.type = type
        # the signal to noise ratio in dB
        self.snr = SNR
        # Convert SNR from dB scale to linear scale
        self.snr_linear = 10 ** (self.snr / 10)
        # Calculate the noise standard deviation
        self.noise_std = np.sqrt(1 / (2 * self.snr_linear))
        print(f'[INFO-CHANNEL] SNR: {self.snr}')
        print(f'[INFO-CHANNEL] Noise std: {self.noise_std}')
        # Set the training device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
    
    def set_snr(self, snr):
        self.snr = snr
        self.snr_linear = 10 ** (self.snr / 10)
        self.noise_std = np.sqrt(1 / (2 * self.snr_linear))

    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
    
    def __call__(self, codeword):
                
        if self.type.upper() == 'AWGN':
            # Generate the noise
            noise = self.noise_std * torch.randn_like(codeword)
            # Add the noise to the input signal
            noisy_codeword = codeword + noise
        else:
            raise ValueError(f"Invalid channel type: {type}. Currently only 'AWGN' is supported.")
        
        return noisy_codeword, noise
    
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
    