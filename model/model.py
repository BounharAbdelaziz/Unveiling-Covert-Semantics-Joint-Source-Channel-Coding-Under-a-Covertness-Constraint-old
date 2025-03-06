import torch
import torch.nn as nn
from model.channel import Channel
from model.semantic_encoder import SemanticEncoder
from model.jscc_encoder import JSCCEncoder
from model.semantic_decoder import SemanticDecoder
from model.classifier import Classifier

import numpy as np

from training_mode import TRAINING_MODE
from options.base_options import Options

# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

class SemanticCovertModel(nn.Module):

    def __init__(self, 
                    training_options: Options = None,                                # The training options
                    TRAIN_MODE: TRAINING_MODE = TRAINING_MODE.FEATURE_EXTRACTOR,     # Which training phase 1 (FEATURE EXTRACTOR and Classifier) or 2 (AUTO ENCODER) or 3 (TEST) or 4 (TEST FEATURE EXTRACTOR only)
                    LOAD_FORCED: bool = False,                                       # Load the model even if the training mode is not TRAINING_MODE.AUTO_ENCODER
                ):
        
        super(SemanticCovertModel, self).__init__()

        # Set the training options
        self.training_options = training_options
        # the channel
        self.channel = Channel(type=training_options.channel_type , SNR=training_options.SNR)
        # Set the training mode
        self.TRAIN_MODE = TRAIN_MODE
        
        print(f'-------------------------------------------------------------------')
        print(f'[INFO] LOAD_FORCED: {LOAD_FORCED}')
        print(f'-------------------------------------------------------------------')
        if LOAD_FORCED:
            # The path to the semantic encoder checkpoint
            SEMANTIC_ENCODER_CKPT_PATH = f'{training_options.checkpoint_directory}/semantic_encoder.pth'
            # The path to the classifier checkpoint
            CLASSIFIER_CKPT_PATH = f'{training_options.checkpoint_directory}/classifier.pth'
            # The path to the JSCC encoder checkpoint             
            JSCC_ENCODER_CKPT_PATH = f'{training_options.checkpoint_directory}/jscc_encoder_forced_last.pth'
            # The path to the semantic decoder checkpoint         
            SEMANTIC_DECODER_CKPT_PATH = f'{training_options.checkpoint_directory}/semantic_decoder_forced_last.pth'
        else:
            if TRAIN_MODE == TRAINING_MODE.TEST and training_options.test_with_joint_model:
                # The path to the semantic encoder checkpoint
                SEMANTIC_ENCODER_CKPT_PATH = f'{training_options.checkpoint_directory}/semantic_encoder_joint.pth'
                # The path to the classifier checkpoint
                CLASSIFIER_CKPT_PATH = f'{training_options.checkpoint_directory}/classifier_joint.pth'
                # The path to the JSCC encoder checkpoint             
                JSCC_ENCODER_CKPT_PATH = f'{training_options.checkpoint_directory}/jscc_encoder_joint.pth'
                # The path to the semantic decoder checkpoint         
                SEMANTIC_DECODER_CKPT_PATH = f'{training_options.checkpoint_directory}/semantic_decoder_joint.pth'
            else:
                # The path to the semantic encoder checkpoint
                SEMANTIC_ENCODER_CKPT_PATH = f'{training_options.checkpoint_directory}/semantic_encoder.pth'
                # The path to the classifier checkpoint
                CLASSIFIER_CKPT_PATH = f'{training_options.checkpoint_directory}/classifier.pth'
                # The path to the JSCC encoder checkpoint             
                JSCC_ENCODER_CKPT_PATH = f'{training_options.checkpoint_directory}/jscc_encoder.pth'
                # The path to the semantic decoder checkpoint         
                SEMANTIC_DECODER_CKPT_PATH = f'{training_options.checkpoint_directory}/semantic_decoder.pth'
        
        if training_options.train_covert_model:
            print(f'[INFO] Training the covert model')
        else:
            print(f'[INFO] Training the non-covert model')
        
        # the size of the semantic vector
        k = training_options.k
        # maximum power scaling factor, we use 1 for simplicity
        A = training_options.A 
        # the maximum power of the transmitted signal
        self.maximum_power = A * np.sqrt(training_options.blocklength)
        # the verbose mode
        self.VERBOSE = training_options.verbose
        # Initialize the semantic encoder and the classifier
        if TRAIN_MODE == TRAINING_MODE.FEATURE_EXTRACTOR:
            # the semantic encoder
            self.semantic_encoder = SemanticEncoder(    training_options=training_options,    
                                                        input_dim=training_options.input_dim_semantic_encoder, 
                                                        hidden_dim=training_options.hidden_dim_semantic_encoder, 
                                                        n_hidden=training_options.n_hidden_semantic_encoder, 
                                                        output_dim=k,
                                                        do_quantize=training_options.do_quantize_U,
                                                    )
            # the classifier
            self.classifier = Classifier(   input_dim=k, 
                                            hidden_dim=training_options.hidden_dim_classifier, 
                                            n_hidden=training_options.n_hidden_classifier, 
                                            output_dim=training_options.output_dim_classifier
                                        )
        
        # in both cases we need to keep the semantic encoder and the classifier, but their weights will be fixed in the TRAINING_MODE.AUTO_ENCODER case
        elif TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER or self.TRAIN_MODE == TRAINING_MODE.TEST:
   
            # the semantic encoder
            self.semantic_encoder = SemanticEncoder(    training_options=training_options,
                                                        input_dim=training_options.input_dim_semantic_encoder, 
                                                        hidden_dim=training_options.hidden_dim_semantic_encoder, 
                                                        n_hidden=training_options.n_hidden_semantic_encoder, 
                                                        output_dim=k,
                                                        do_quantize=training_options.do_quantize_U,
                                                    )
            
            # the joint source channel encoder
            self.jscc_encoder = JSCCEncoder(input_dim=k, 
                                            hidden_dim=training_options.hidden_dim_jscc_encoder ,
                                            n_hidden=training_options.n_hidden_jscc_encoder ,
                                            output_dim=training_options.output_dim_jscc_encoder,
                                            do_quantize=training_options.do_quantize,
                                            )

            # the semantic decoder
            self.semantic_decoder = SemanticDecoder(    input_dim=training_options.input_dim_jscc_decoder ,
                                                        hidden_dim=training_options.hidden_dim_jscc_decoder ,
                                                        n_hidden=training_options.n_hidden_jscc_decoder ,
                                                        output_dim=k,
                                                    )
            
            # the classifier
            self.classifier = Classifier(   input_dim=k, 
                                            hidden_dim=training_options.hidden_dim_classifier, 
                                            n_hidden=training_options.n_hidden_classifier, 
                                            output_dim=training_options.output_dim_classifier
                                        )
                
        # Initialize the semantic encoder and the classifier
        elif TRAIN_MODE == TRAINING_MODE.TEST_FEATURE_EXTRACTOR:
            # the semantic encoder
            self.semantic_encoder = SemanticEncoder(    training_options=training_options,    
                                                        input_dim=training_options.input_dim_semantic_encoder, 
                                                        hidden_dim=training_options.hidden_dim_semantic_encoder, 
                                                        n_hidden=training_options.n_hidden_semantic_encoder, 
                                                        output_dim=k,
                                                        do_quantize=training_options.do_quantize_U,
                                                    )
            # the classifier
            self.classifier = Classifier(   input_dim=k, 
                                            hidden_dim=training_options.hidden_dim_classifier, 
                                            n_hidden=training_options.n_hidden_classifier, 
                                            output_dim=training_options.output_dim_classifier
                                        )
        
        if self.VERBOSE:
            print(f'-------------------------------------------------------------------')
            print(f'[INFO] Semantic Encoder: {self.semantic_encoder}')
            print(f'[INFO] jscc encoder: {self.jscc_encoder}')
            print(f'[INFO] Semantic Decoder: {self.semantic_decoder}')
            print(f'[INFO] Classifier: {self.classifier}')
            print(f'-------------------------------------------------------------------')
            
        # loading and freezing the pre-trained weights of the semantic encoder and the classifier
        if TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER and training_options.dataset.upper() != 'IID':
            # load the weights of the semantic encoder and the classifier
            self.semantic_encoder.load_state_dict(torch.load(SEMANTIC_ENCODER_CKPT_PATH))
            self.classifier.load_state_dict(torch.load(CLASSIFIER_CKPT_PATH))
            # freeze the weights of the semantic encoder and the classifier
            for param in self.semantic_encoder.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = False
                
            print(f'-------------------------------------------------------------------')
            print(f'[INFO] Semantic Encoder loaded from: {SEMANTIC_ENCODER_CKPT_PATH}')
            print(f'[INFO] Classifier loaded from: {CLASSIFIER_CKPT_PATH}')
            print(f'[INFO] The weights of the Semantic Encoder and the Classifier are loaded and frozen.')
            print(f'-------------------------------------------------------------------')
            
        # loading the weights of the semantic encoder, the JSCC encoder, the semantic decoder and the classifier
        elif TRAIN_MODE == TRAINING_MODE.TEST:
            if training_options.dataset.upper() != 'IID':
                semantic_encoder_weights = torch.load(SEMANTIC_ENCODER_CKPT_PATH)
                jscc_encoder_weights = torch.load(JSCC_ENCODER_CKPT_PATH)
                semantic_decoder_weights = torch.load(SEMANTIC_DECODER_CKPT_PATH)
                classifier_weights = torch.load(CLASSIFIER_CKPT_PATH)
                self.semantic_encoder.load_state_dict(semantic_encoder_weights)
                self.jscc_encoder.load_state_dict(jscc_encoder_weights)
                self.semantic_decoder.load_state_dict(semantic_decoder_weights)
                self.classifier.load_state_dict(classifier_weights)
                
                print(f'-------------------------------------------------------------------')
                print(f'[INFO-TEST] Semantic Encoder loaded from: {SEMANTIC_ENCODER_CKPT_PATH}')
                print(f'[INFO-TEST] JSCC Encoder loaded from: {JSCC_ENCODER_CKPT_PATH}')
                print(f'[INFO-TEST] Semantic Decoder loaded from: {SEMANTIC_DECODER_CKPT_PATH}')
                print(f'[INFO-TEST] Classifier loaded from: {CLASSIFIER_CKPT_PATH}')
                print(f'-------------------------------------------------------------------')
            
            else:
                # in IID dataset, the semantic vectors are already provided, so no need to extract them
                self.jscc_encoder.load_state_dict(torch.load(JSCC_ENCODER_CKPT_PATH))
                self.semantic_decoder.load_state_dict(torch.load(SEMANTIC_DECODER_CKPT_PATH))
                
                print(f'-------------------------------------------------------------------')
                print(f'[INFO-TEST] JSCC Encoder loaded from: {JSCC_ENCODER_CKPT_PATH}')
                print(f'[INFO-TEST] Semantic Decoder loaded from: {SEMANTIC_DECODER_CKPT_PATH}')
                print(f'-------------------------------------------------------------------')
                
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
        
    def forward(self, x):

        if self.training_options.dataset.upper() == 'IID':
            
            # apply joint source channel coding to the sequence U^k
            encoded_information = self.jscc_encoder(x)
            
            # add noise to the encoded information
            received_information, noise = self.channel(encoded_information)
            
            # decode the received information
            decoded_semantic_information = self.semantic_decoder(received_information)
            
            
            return x, encoded_information, decoded_semantic_information, noise
        else:
            if self.TRAIN_MODE == TRAINING_MODE.FEATURE_EXTRACTOR:
                # first extract the semantic information from the input
                semantic_information = self.semantic_encoder(x)
                # predict the class of the decoded information
                logits = self.classifier(semantic_information)
                
                return logits
            
            # in both cases we need to keep the semantic encoder and the classifier, but their weights will be fixed in the TRAINING_MODE.AUTO_ENCODER case
            elif self.TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER or self.TRAIN_MODE == TRAINING_MODE.TEST:
                
                # first extract the semantic information from the input
                semantic_information = self.semantic_encoder(x)
                
                # apply joint source channel coding
                encoded_information = self.jscc_encoder(semantic_information)
                
                # add noise to the encoded information
                received_information, noise = self.channel(encoded_information)
                
                # decode the received information
                decoded_semantic_information = self.semantic_decoder(received_information)
                
                # get the logits of the decoded information to predict the class
                logits = self.classifier(decoded_semantic_information)
                
                return semantic_information, encoded_information, decoded_semantic_information, logits, noise
    
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
    
    