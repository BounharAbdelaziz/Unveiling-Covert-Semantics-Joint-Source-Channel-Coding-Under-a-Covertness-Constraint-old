import torch.nn as nn
import torch.nn.init as init

class NN(nn.Module):
    
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
    
    def __init__(self, ):
        super(NN, self).__init__()

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