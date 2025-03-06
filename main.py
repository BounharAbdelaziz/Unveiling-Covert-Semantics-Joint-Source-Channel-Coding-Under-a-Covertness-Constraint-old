from training_mode import TRAINING_MODE
from model.trainer import Trainer
import numpy as np
from options.base_options import Options



if __name__ == "__main__":

    # Training phase; 1 (FEATURE_EXTRACTOR) or 2 (AUTO_ENCODER)
    TRAIN_MODE = TRAINING_MODE.FEATURE_EXTRACTOR
    # TRAIN_MODE = TRAINING_MODE.AUTO_ENCODER

    training_options = Options(TRAIN_MODE).parse()

    # the flag to train the covert model
    TRAIN_COVERT_MODEL = training_options.train_covert_model
   
    # max power constraint
    if TRAIN_COVERT_MODEL:
        print(f'[INFO] Training the covert model with k = O(sqrt(n))')
        maximum_power = training_options.A * np.sqrt(training_options.blocklength * training_options.epsilon_n)
    else:
        lambda_power_constraint = training_options.lambda_power_constraint
        maximum_power = training_options.A * np.sqrt(training_options.blocklength)
        print(f'[INFO] Training the covert model with k = O(n)')
    
    print('[INFO] Initializing the trainer')
    # Create the trainer     
    trainer = Trainer( training_options,  # the training options
                        maximum_power,    # the maximum power constraint
                        TRAIN_MODE,       # Which model to train Train the model jointly or separately
            )
    
    print(f'[INFO] Model architecture :{trainer.semantic_covert_model}')

    # start training
    trainer.train(training_options.n_epochs)