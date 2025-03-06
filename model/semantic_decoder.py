import torch.nn as nn
from model.quantizer import StraightThroughEstimator
from model.neural_network import NN

class SemanticDecoder(NN):

    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #

    def __init__(self, 
                    input_dim=2048,         # Number of input features (the blocklength of the code)
                    hidden_dim=2048,        # Number of hidden units in each layer
                    n_hidden=3,             # Number of hidden layers
                    output_dim=10,          # Number of ouput features (dimension of the semantic vector)
                    do_quantize=False,      # Whether to quantize the output (the semantic vector)
                ):
        
        super(SemanticDecoder, self).__init__()

        self.do_quantize = do_quantize

        # Define the layers of the semantic decoder
        layers = []
        if n_hidden == 1:
            layers.append(nn.Linear(input_dim, output_dim))

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

        # Stack the layers in a sequential model
        self.semantic_decoder = nn.Sequential(*layers)

        # if we want to quantize the output
        if self.do_quantize:
            self.quantizer = StraightThroughEstimator()
        
        # Initialize the weights
        self.init_weights(self.semantic_decoder)

    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
        
    def forward(self, x):

        # Pass the input through the semantic decoder to get the decoded semantic information
        decoded_semantic_information = self.semantic_decoder(x)
        
        # if we want to quantize the output
        if self.do_quantize:
            decoded_quantized_information = self.quantizer(decoded_semantic_information)

            return decoded_quantized_information
        
        return decoded_semantic_information
    
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #