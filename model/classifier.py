import torch.nn as nn
from model.neural_network import NN

class Classifier(NN):
    
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
    
    def __init__(self, 
                    input_dim=128,      # Number of input features (dimension of the semantic vector)
                    hidden_dim=256,     # Number of hidden units
                    n_hidden=1,         # Number of hidden layers
                    output_dim=10,      # Number of classes
                ):
        super(Classifier, self).__init__()

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
        self.classifier = nn.Sequential(*layers)
        
        # Initialize the weights
        self.init_weights(self.classifier)
        
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
        
    def forward(self, x):
        logits = self.classifier(x)
        return logits
    
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #