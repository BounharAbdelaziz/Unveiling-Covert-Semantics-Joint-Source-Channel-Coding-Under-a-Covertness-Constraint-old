import argparse
import os
import numpy as np
import random

import torch
from training_mode import TRAINING_MODE

class Options():
    """
    This class is used to store the options of the experiment. 
    It is also used to parse the arguments passed to the experiment.
    Attributes:
        parser (argparse.ArgumentParser): the parser object.
        train_mode (bool): if set to True, the experiment is in training mode. Otherwise, it is in testing mode.
    """
    
    # -------------------------------------------------------------------------------------------- #
    
    def __init__(self, TRAINING_MODE: TRAINING_MODE = TRAINING_MODE.TEST) -> None:
        """Initializes the Options class which takes care of initializing all hyperparameters of the experiment."""
        # initialize the parser  
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)        

        # learning parameters
        self.parser.add_argument('--train_covert_model', action='store_true', help='if specified, trains the covert model with k = O(sqrnt(n))')
        self.parser.add_argument('--batch_size', type=int, default=128, required=False, help='the batch size.')
        self.parser.add_argument('--learning_rate', type=float, default=5e-3, required=False, help='the learning rate.')
        self.parser.add_argument('--optimizer', type=str, default='Adam', required=False, help='the optimizer.')
        self.parser.add_argument('--n_epochs', type=int, default=10, required=False, help='the number of episodes in the training phase.')
        self.parser.add_argument('--reconstruction_loss_fct', type=str, default='L2', required=False, help='the loss function to use for the reconstruction loss.')
        self.parser.add_argument('--lambda_reconstruction', type=float, default=1, required=False, help='the weight of the reconstruction loss.')
        self.parser.add_argument('--lambda_classification', type=float, default=1, required=False, help='the the weight of the classification loss.')
        self.parser.add_argument('--lambda_power_constraint', type=float, default=0.8, required=False, help='the weight of the power constraint loss.')
        self.parser.add_argument('--lambda_covertness_constraint', type=float, default=0.01, required=False, help='the weight of the covertness loss.')
        self.parser.add_argument('--lambda_l2_regularization', type=float, default=0.005, required=False, help='the weight of the L2 regularization loss of the AE model.')
        self.parser.add_argument('--epsilon_n', type=float, default=0.01, required=False, help='the epsilon muliplying sqrt n.')
        self.parser.add_argument('--target_kl_div', type=float, required=True, help='the targeted covertness constraint.')
        self.parser.add_argument('--dataset', type=str, default='MNIST', required=False, help='the dataset to train on [MNIST | CIFAR10 | IID].')
        self.parser.add_argument('--dataset_path', type=str, default='./datasets/', required=False, help='the dataset path.')
        self.parser.add_argument('--n_samples_train', type=int, default=50000, required=False, help='the generated training dataset size.')
        self.parser.add_argument('--n_samples_test', type=int, default=10000, required=False, help='the generated testing dataset size.')
        self.parser.add_argument('--proba', type=float, default=0.65, required=False, help='the probability of observing a 1 in the Bernoulli distribution. (used in IID dataset only)')
        self.parser.add_argument('--k', type=int, required=False, help='the size of the semantic feature vector.')
        self.parser.add_argument('--use_delta_n_in_k', action='store_true', help='to make k = sqrt(n*delta_n) instead of sqrt(n).')
        self.parser.add_argument('--use_sk', action='store_true', help='use a secret-key to allow for local randomness at the encoder and decoder.')

        # communication parameters
        self.parser.add_argument('--blocklength', type=int, default=1024, required=False, help='the communication blocklength.')
        self.parser.add_argument('--A', type=int, default=1, required=False, help='the constant A influences the power constraint, the number of 1\'s in the codeword x^n')
        self.parser.add_argument('--C', type=int, default=1, required=False, help='the constant C influences the k parameter, i.e. k = C * sqrt(n)')
        self.parser.add_argument('--SNR', type=float, default=10, required=False, help='The SNR level in dBm/Hz.')
        self.parser.add_argument('--channel_type', type=str, default='AWGN', required=False, help='The channel type.')

        # model parameters

        # semantic encoder
        self.parser.add_argument('--input_dim_semantic_encoder', type=int, default=28*28, required=False, help='the dimension of the input.')
        self.parser.add_argument('--hidden_dim_semantic_encoder', type=int, default=64, required=False, help='the dimension of the hidden layer in the semantic encoder.')
        self.parser.add_argument('--n_hidden_semantic_encoder', type=int, default=1, required=False, help='the number of hidden layers in the semantic encoder.')
        self.parser.add_argument('--output_dim_semantic_encoder', type=int, default=1024, required=False, help='the dimension of the input.')
        
        # JSCC encoder
        self.parser.add_argument('--input_dim_jscc_encoder', type=int, default=1024, required=False, help='the dimension of the input.')
        self.parser.add_argument('--hidden_dim_jscc_encoder', type=int, default=32, required=False, help='the dimension of the hidden layer in the JSCC encoder.')
        self.parser.add_argument('--n_hidden_jscc_encoder', type=int, default=2, required=False, help='the number of hidden layers in the JSCC encoder.')
        self.parser.add_argument('--output_dim_jscc_encoder', type=int, default=1024, required=False, help='the dimension of the input.')
        self.parser.add_argument('--do_quantize', type=int, default=0, required=False, help='if we want to add a quantization of the output of the JSCC encoder.')
        self.parser.add_argument('--do_quantize_U', type=int, default=0, required=False, help='if we want to add a quantization of the output of the Semantic encoder.')
        self.parser.add_argument('--do_quantize_U_hat', type=int, default=0, required=False, help='if we want to add a quantization of the output of the decoder.')
        self.parser.add_argument('--quantization_levels', type=int, nargs='*', default=[0,1], required=False, help='the quantization levels at the output of the JSCC encoder.')

        # JSCC decoder
        self.parser.add_argument('--input_dim_jscc_decoder', type=int, default=1024, required=False, help='the dimension of the input.')
        self.parser.add_argument('--hidden_dim_jscc_decoder', type=int, default=32, required=False, help='the dimension of the hidden layer in the JSCC encoder.')
        self.parser.add_argument('--n_hidden_jscc_decoder', type=int, default=2, required=False, help='the number of hidden layers in the JSCC encoder.')
        self.parser.add_argument('--output_dim_jscc_decoder', type=int, default=1024, required=False, help='the dimension of the input.')
        
        # classifier
        self.parser.add_argument('--input_dim_classifier', type=int, default=1024, required=False, help='the dimension of the input.')
        self.parser.add_argument('--hidden_dim_classifier', type=int, default=32, required=False, help='the dimension of the hidden layer in the classifier.')
        self.parser.add_argument('--n_hidden_classifier', type=int, default=1, required=False, help='the number of hidden layers in the classifier.')
        self.parser.add_argument('--output_dim_classifier', type=int, default=10, required=False, help='the dimension of the output.')
        
        # experiment parameters
        self.parser.add_argument('--seed', type=int, default=1998, required=False, help='seed for reproducibility.')

        # additional parameters
        self.parser.add_argument('--verbose', action='store_true', help='if specified, set to debug mode')
        self.parser.add_argument('--log_tensorboard', action='store_true', help='if specified, log the training process to tensorboard')
        self.parser.add_argument('--continue_training', action='store_true', help='if specified, continue training from a checkpoint.')
        self.parser.add_argument('--load_checkpoint', type=str, required=False, help='indicates the checkpoint to load and continue training from.')

        # testing parameters
        self.parser.add_argument('--MAX_TEST_SAMPLES', type=int, default=1000, help='the number of samples in the testing phase.')
        self.parser.add_argument('--test_mode', action='store_true', default=False, help='if specified, run the testing phase.')
        self.parser.add_argument('--test_classifier_only', action='store_true', default=False, help='if specified, tests only the classifier.')
        self.parser.add_argument('--test_with_joint_model', action='store_true', default=False, help='if specified, tests the model after the joint training phase.')
        self.parser.add_argument('--test_reconstruction_only', action='store_true', default=False, help='if specified, tests the reconstruction model only.')

        self.TRAINING_MODE = TRAINING_MODE
        
    # -------------------------------------------------------------------------------------------- #
        
    def parse(self) -> argparse.Namespace:
        """Parses and saves (to a file) the arguments of the experiment.
        
        Returns:
            argparse.Namespace: the options of the experiment.
        """
        
        # parse the arguments
        options = self.parser.parse_args()
        
        # fix the seed for reproducibility
        self.fix_seed(options.seed)

        if options.train_covert_model:
            # set k = O(sqrt(n))
            # k = np.floor(options.C*np.sqrt(options.blocklength)).astype(int)

            # check if k is set in the arguments
            if hasattr(options, 'k') and options.k is not None:
                print(f"[INFO] Using the specified k = {options.k}")
                k = options.k
            else:
                print(f"[INFO] Using the default k = C * sqrt(n*delta_n)")
                if options.use_delta_n_in_k:
                    k = np.floor(options.C*np.sqrt(options.blocklength*options.delta_n)).astype(int) # consider the covertness constraint
                else:
                    k = np.floor(options.C*np.sqrt(options.blocklength)).astype(int)
                options.k = k

            options.input_dim_jscc_encoder = k
            options.ouput_dim_jscc_decoder = k
  
        else:
            # set k = O(n)

             # check if k is set in the arguments
            if hasattr(options, 'k') and options.k is not None:
                print(f"[INFO] Using the specified k = {options.k}")
                k = options.k
            else:
                print(f"[INFO] Using the default k = C * n")
                k = options.C * options.blocklength
                options.k = k

            options.input_dim_jscc_encoder = k
            options.ouput_dim_jscc_decoder = k

        # create the checkpoint directory path so as to organize the savings
        if options.train_covert_model:
            model_type = 'covert' 
        else:
            if options.lambda_power_constraint:
                model_type = 'non-covert-KL-training'
            else:
                model_type = 'non-covert'

        print(f"[INFO-options] Model type: {model_type}")
                
        if options.do_quantize:
            is_qantized_enc = 'quantized-enc'
        else:
            is_qantized_enc = 'not-quantized-enc'
        
        if options.do_quantize_U:
            is_qantized_sem = 'quantized-sem'
        else:
            is_qantized_sem = 'not-quantized-sem'

        if options.do_quantize_U_hat:
            is_qantized_dec = 'quantized-dec'
        else:
            is_qantized_dec = 'not-quantized-dec'
        
        if options.use_sk:
            uses_secret_key = 'with-sk'
        else:
            uses_secret_key = 'without-sk'
            
        # create the checkpoint directory path so as to organize the savings
        if hasattr(options, 'k') and options.k is not None:
            options.checkpoint_directory = f'./checkpoints/{uses_secret_key}/{options.dataset.upper()}/n/{options.blocklength}/{model_type}/{is_qantized_sem}/{is_qantized_enc}/{is_qantized_dec}/C/{options.C}/A/{options.A}/k/{options.k}/SNR/{str(options.SNR)}_db/'
            options.logging_directory = f'./logs/{uses_secret_key}/{options.dataset.upper()}/n/{options.blocklength}/{model_type}/{is_qantized_sem}/{is_qantized_enc}/{is_qantized_dec}/C/{options.C}/A/{options.A}/k/{options.k}/SNR/{str(options.SNR)}_db/'

        else:
            options.checkpoint_directory = f'./checkpoints/{uses_secret_key}/{options.dataset.upper()}/n/{options.blocklength}/{model_type}/{is_qantized_sem}/{is_qantized_enc}/{is_qantized_dec}/C/{options.C}/A/{options.A}/SNR/{str(options.SNR)}_db/'
            options.logging_directory = f'./logs/{uses_secret_key}/{options.dataset.upper()}/n/{options.blocklength}/{model_type}/{is_qantized_sem}/{is_qantized_enc}/{is_qantized_dec}/C/{options.C}/A/{options.A}/SNR/{str(options.SNR)}_db/'
                
        # Create the directories if they do not exist
        if not os.path.exists(options.checkpoint_directory):
            print(f"Creating checkpoint directory at {options.checkpoint_directory}")
            os.makedirs(os.path.dirname(options.checkpoint_directory), exist_ok=True)

        # print the options
        self.print_options(options)
        # save the options to a file if not in test mode
        if not options.test_mode:
            self.save_options(options)

        return options  
    
    # -------------------------------------------------------------------------------------------- #
    
    def save_options(self, options) -> None:
        """Saves the options of the experiment to a file.
        
        Returns:
            None
        """
        saving_directory = options.checkpoint_directory
        # is_training = TRAINING_MODE != TRAINING_MODE.TEST
        if self.TRAINING_MODE == TRAINING_MODE.FEATURE_EXTRACTOR:
            filename = os.path.join(saving_directory, 'FE_training_options.txt')
        elif self.TRAINING_MODE == TRAINING_MODE.AUTO_ENCODER:
            filename = os.path.join(saving_directory, 'AE_training_options.txt')
        elif self.TRAINING_MODE == TRAINING_MODE.TEST:
            filename =  os.path.join(saving_directory, 'testing_options.txt')
                
        
        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)
            print(f"Creating checkpoint directory at {saving_directory}")
        
        print(f"Saving options in file at {filename}")
        
        with open(filename, 'wt') as file:
            # get the string representation of the options
            options_values = self.__str__(options)
            file.write(options_values)      
    
    # -------------------------------------------------------------------------------------------- #
    
    def print_options(self, options: argparse.Namespace) -> None:
        """Prints the options of the experiment.
        
        Args:
            options (argparse.Namespace): the options of the experiment.
        
        Returns:
            None
        """
        # get the string representation of the options
        options_values = self.__str__(options)
        # print the options
        print(options_values)
    
    # -------------------------------------------------------------------------------------------- #
    
    def fix_seed(self, seed: int) -> None:
        """
        Sets a random seed for reproducibility
        
        Args:
            seed (int): The seed to use for randomness.
        
        Returns:
            None
        """
        
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you're using GPU
        
        # Additional settings for deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -------------------------------------------------------------------------------------------- #
    
    def __str__(self, options: argparse.Namespace) -> str:
        """Returns a string representation of the options.
        
        Returns:
            str: the string representation of the options.
        """
        
        options_values = ''
        options_values += '-------------------------------------------- Options ---------------------------------------------\n'
        for k, v in sorted(vars(options).items()):
            changed_default_value = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                changed_default_value = '\t[default: %s]' % str(default_value)
            options_values += f'{str(k):>30}: {str(v):<10}{changed_default_value}\n'
        options_values += '--------------------------------------------------------------------------------------------------'
        return options_values
    
    # -------------------------------------------------------------------------------------------- #