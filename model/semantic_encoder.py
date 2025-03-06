import torch.nn as nn
import torch.nn.init as init
from model.quantizer import StraightThroughEstimator

class SemanticEncoder(nn.Module):
    def __init__(self,
                    training_options=None,  # Training options
                    input_dim=1,            # Number of input features (the number of channels of the image)
                    output_channels=32,     # Number of output channels in the CNN 
                    hidden_dim=256,         # Number of hidden units in the MLP
                    n_hidden=2,             # Number of hidden layers in the MLP
                    output_dim=10,          # Number of ouput features (dimension of the semantic vector)
                    do_quantize=True,       # Whether to quantize the output (the semantic vector)
                ):
        
        super(SemanticEncoder, self).__init__()


        # CNN layers, to extract features from the input
        cnn_layers = []
        # Linear layers, to process the features extracted by the CNN
        mlp_layers = []
        
        # --------------------------------------    CNN     -------------------------------------------- #
        
        # we first define the CNN layers depending on the dataset
        if training_options.dataset.upper() == 'MNIST':
            cnn_layers.append(nn.Conv2d(1, output_channels, kernel_size=3, stride=1, padding=1))
            cnn_layers.append(nn.BatchNorm2d(output_channels))
            cnn_layers.append(nn.Dropout2d(p=0.1))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d((2, 2)))
            cnn_layers.append(nn.Flatten())
            
            # dimension of the input to the MLP
            dim = 14*14*output_channels
            
            
        elif training_options.dataset.upper() == 'CIFAR10':
            cnn_layers.append(nn.Conv2d(3, output_channels, kernel_size=3, stride=1, padding=1))
            cnn_layers.append(nn.BatchNorm2d(output_channels))
            cnn_layers.append(nn.Dropout2d(p=0.1))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d((2, 2)))
            cnn_layers.append(nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1))
            cnn_layers.append(nn.BatchNorm2d(output_channels))
            cnn_layers.append(nn.Dropout2d(p=0.1))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool2d((2, 2)))
            cnn_layers.append(nn.Flatten())
            
            # dimension of the input to the MLP
            dim = 8*8*output_channels
            
        elif training_options.dataset.upper() == 'IDD':
            
            cnn_layers.append(nn.Linear(input_dim, hidden_dim))  
            cnn_layers.append(nn.BatchNorm1d(hidden_dim))
            cnn_layers.append(nn.Dropout2d(p=0.1))
            cnn_layers.append(nn.LeakyReLU())
            
            dim = hidden_dim
            
        # Stack the layers in a sequential model
        self.semantic_encoder_cnn = nn.Sequential(*cnn_layers)

        # --------------------------------------    MLP     -------------------------------------------- #
                
        if n_hidden == 1:
            mlp_layers.append(nn.Linear(dim, output_dim))
            mlp_layers.append(nn.LeakyReLU())

        else:
            for i in range(n_hidden-1):
                if i == 0:
                    mlp_layers.append(nn.Linear(dim, hidden_dim))  
                else:
                    mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
                mlp_layers.append(nn.Dropout1d(p=0.1))
                mlp_layers.append(nn.BatchNorm1d(hidden_dim))
                mlp_layers.append(nn.LeakyReLU())

            # last layer
            mlp_layers.append(nn.Linear(hidden_dim, output_dim))
            mlp_layers.append(nn.LeakyReLU())
                
        # Stack the layers in a sequential model
        self.semantic_encoder_mlp = nn.Sequential(*mlp_layers)

        # if we want to quantize the output
        self.do_quantize = do_quantize
        if self.do_quantize:
            self.quantizer = StraightThroughEstimator()
        
        # Initialize the weights        
        self.init_weights(self.semantic_encoder_cnn)
        self.init_weights(self.semantic_encoder_mlp)
        
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
        
    def forward(self, x):
        
        semantic_encoder_cnn = self.semantic_encoder_cnn(x)
        semantic_encoder_mlp = self.semantic_encoder_mlp(semantic_encoder_cnn)

        # if we want to quantize the output
        if self.do_quantize:
            quantized_information = self.quantizer(semantic_encoder_mlp)

            return quantized_information
        
        return semantic_encoder_mlp
    
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
    
    def init_weights(self, model):
        """
        Initialize weights for all layers in the model using Kaiming uniform initialization.
        
        Args:
            model: The PyTorch model for which weights are to be initialized.
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # Optionally, initialize bias to zeros
                    
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #