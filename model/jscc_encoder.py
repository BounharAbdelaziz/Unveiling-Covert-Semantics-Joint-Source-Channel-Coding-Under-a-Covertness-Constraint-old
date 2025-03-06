import torch.nn as nn
from model.neural_network import NN
from model.quantizer import StraightThroughEstimator
    

class JSCCEncoder(NN):
    def __init__(self, 
                    input_dim=10,       # Number of input features (dimension of the semantic vector)
                    hidden_dim=2048,    # Number of hidden units in each layer
                    n_hidden=2,         # Number of hidden layers
                    output_dim=2048,    # Number of output features (the blocklength of the code)
                    do_quantize=False,  # Whether to quantize the output (the codeword)
                ):
        
        super(JSCCEncoder, self).__init__()

        layers = []
        
        if n_hidden == 1:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim))

        else:
            for i in range(n_hidden-1):
                if i == 0:
                    layers.append(nn.Linear(input_dim, hidden_dim))
                else:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.Dropout1d(p=0.1))
                layers.append(nn.LeakyReLU())

            # last layer
            layers.append(nn.Linear(hidden_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim))
            
        # Stack the layers in a sequential model
        self.jscc_encoder = nn.Sequential(*layers)
        
        # if we want to quantize the output
        self.do_quantize = do_quantize
        if self.do_quantize:
            self.quantizer = StraightThroughEstimator()
        
        # Initialize the weights
        self.init_weights(self.jscc_encoder)
        
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
        
    def forward(self, x):

        # Pass the input through the semantic encoder to get the codeword
        encoded_information = self.jscc_encoder(x)
        
        # if we want to quantize the output
        if self.do_quantize:
            quantized_information = self.quantizer(encoded_information)
            return quantized_information
        
        return encoded_information

    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #