import torch
from model.model import SemanticCovertModel
from model.loss import ReconstructionLoss, ClassificationLoss, PowerConstraintLoss, CovertnessConstraintLoss
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from training_mode import TRAINING_MODE
from options.base_options import Options
from datasets.iid_dataset import IIDDataset
import numpy as np
from scipy.stats import entropy

    
class Trainer():

    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
        
    def __init__(self,
                    training_options: Options =None,                                # the training options 
                    maximum_power=1.0,                                              # the maximum power constraint
                    TRAIN_MODE: TRAINING_MODE = TRAINING_MODE.FEATURE_EXTRACTOR,    # Which training phase 1 (FEATURE EXTRACTOR and Classifier) or 2 (AUTO ENCODER) or 3 (TEST) or 4 (TEST FEATURE EXTRACTOR only)
                    LOAD_FORCED: bool = False,                                      # Load the model from the checkpoint
                ) -> None:
        
        # Set the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Set the training mode
        self.TRAIN_MODE = TRAIN_MODE
        # Set tensorboard logging level
        self.DO_LOG_TENSORBOARD = training_options.log_tensorboard
        # Checkpoint paths
        # The path to the semantic encoder checkpoint
        self.SEMANTIC_ENCODER_CKPT_SAVING_PATH = f'{training_options.checkpoint_directory}/semantic_encoder.pth'
        # The path to the classifier checkpoint
        self.CLASSIFIER_CKPT_SAVING_PATH = f'{training_options.checkpoint_directory}/classifier.pth'
        # The path to the JSCC encoder checkpoint             
        self.JSCC_ENCODER_CKPT_SAVING_PATH = f'{training_options.checkpoint_directory}/jscc_encoder.pth'
        # The path to the semantic decoder checkpoint         
        self.SEMANTIC_DECODER_CKPT_SAVING_PATH = f'{training_options.checkpoint_directory}/semantic_decoder.pth'

        # the training options
        self.training_options = training_options
        # the learning rate  
        learning_rate=training_options.learning_rate
        # the batch size  
        batch_size=training_options.batch_size
        # the reconstruction loss function to use
        reconstruction_loss_fct = training_options.reconstruction_loss_fct
        # Define the model
        self.semantic_covert_model = SemanticCovertModel(   training_options,
                                                            TRAIN_MODE, 
                                                            LOAD_FORCED=LOAD_FORCED,
                                                        ).to(self.device)
        # Count the number of trainable parameters
        n_trainable_params = sum(p.numel() for p in self.semantic_covert_model.parameters() if p.requires_grad)
        print(f'-------------------------------------------------------------------')
        print(f'[INFO] Number of trainable parameters: {n_trainable_params}')
        print(f'-------------------------------------------------------------------')
        
        # Define the loss functions depending on the training phase
        # first phase: feature extractor and classifier
        if TRAIN_MODE == TRAINING_MODE.FEATURE_EXTRACTOR:
            self.classification_loss = ClassificationLoss().to(self.device)
        
        # second phase: autoencoder
        elif TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER:
            self.reconstruction_loss = ReconstructionLoss(reconstruction_loss_fct).to(self.device)
            self.power_constraint_loss = PowerConstraintLoss(maximum_power).to(self.device)
            self.covertness_constraint_loss = CovertnessConstraintLoss(delta_n=training_options.epsilon_n).to(self.device)
        
        # at test, if we want to evaluate the losses
        elif TRAIN_MODE == TRAINING_MODE.TEST:
            self.reconstruction_loss = ReconstructionLoss(reconstruction_loss_fct).to(self.device)
            self.classification_loss = ClassificationLoss().to(self.device)
            self.power_constraint_loss = PowerConstraintLoss(maximum_power).to(self.device)
            self.covertness_constraint_loss = CovertnessConstraintLoss(delta_n=training_options.epsilon_n).to(self.device)
        else:
            raise ValueError(f'Invalid TRAIN_MODE: {TRAIN_MODE}. Choose from {TRAINING_MODE.FEATURE_EXTRACTOR}, {TRAINING_MODE.AUTO_ENCODER}, {TRAINING_MODE.TEST}')
        
        # Define the weights for the loss functions
        self.lambda_reconstruction = training_options.lambda_reconstruction
        self.lambda_classification = training_options.lambda_classification
        self.lambda_power_constraint = training_options.lambda_power_constraint
        self.lambda_covertness_constraint = training_options.lambda_covertness_constraint
        self.lambda_l2_regularization = training_options.lambda_l2_regularization
        
        # Define the optimizer
        if training_options.optimizer.upper() == 'SGD':
            self.optimizer = torch.optim.SGD(self.semantic_covert_model.parameters(), lr=learning_rate, momentum=0.9)
        elif training_options.optimizer.upper() == 'ADAM':
            self.optimizer = torch.optim.Adam(self.semantic_covert_model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f'Invalid optimizer: {training_options.optimizer}. Choose from [SGD | ADAM]')

        # The transformations to be applied to the dataset
        print(f'-------------------------------------------------------------------')
        if training_options.dataset.upper() == 'MNIST':
            print(f'[INFO] Loading the MNIST dataset...')
            transform = transforms.Compose([    transforms.ToTensor(),                            # Convert PIL image or numpy.ndarray to tensor
                                                transforms.Normalize((0.1307,), (0.3081,)),       # Normalize the data
                                            ]) 
            
            training_set = torchvision.datasets.MNIST(root=training_options.dataset_path, train=True, download=True, transform=transform)
            testing_set = torchvision.datasets.MNIST(root=training_options.dataset_path, train=False, download=True, transform=transform)
            
            
        elif training_options.dataset.upper() == 'CIFAR10':
            print(f'[INFO] Loading the CIFAR10 dataset...')
            transform = transforms.Compose([    transforms.ToTensor(),                                                                                  # Convert PIL image or numpy.ndarray to tensor
                                                transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768)),       # Normalize the data
                                            ])  
            training_set = torchvision.datasets.CIFAR10(root=training_options.dataset_path, train=True, download=True, transform=transform)
            testing_set = torchvision.datasets.CIFAR10(root=training_options.dataset_path, train=False, download=True, transform=transform)
        
        elif training_options.dataset.upper() == 'IID':
            print(f'[INFO] Loading the IID dataset...')
            # Load the IID dataset
            if training_options.train_covert_model:
                # use the value of k provided in the training options
                if hasattr(training_options, 'k') and training_options.k is not None:
                    k = training_options.k
                    print(f'[INFO-IID] Using the value of k provided in the training options, k = {k}')
                
                # consider the covertness constraint in k
                elif training_options.use_delta_n_in_k:
                    k = np.floor(training_options.C*np.sqrt(training_options.blocklength*training_options.epsilon_n)).astype(int)
                    print(f'[INFO-IID] Considering the covertness constraint in k, k = {k}')
                
                # use k as O(sqrt(n))
                else:
                    k = np.floor(training_options.C*np.sqrt(training_options.blocklength)).astype(int)
                    print(f'[INFO-IID] k is in O(sqrt(n)), k = {k}')
            else:
                if hasattr(training_options, 'k') and training_options.k is not None:
                    # use the value of k provided in the training options
                    k = training_options.k
                    print(f'[INFO-IID] Using the value of k provided in the training options, k = {k}')
                else:
                    # use k as O(n)
                    k = training_options.C * training_options.blocklength
                    print(f'[INFO-IID] k is in O(n), k = {k}')
                
            training_set = IIDDataset(k=k, n_samples=training_options.n_samples_train, p=training_options.proba)
            testing_set = IIDDataset(k=k, n_samples=training_options.n_samples_test, p=training_options.proba)

        else:
            raise ValueError(f'Invalid dataset: {training_options.dataset}. Choose from MNIST, CIFAR10')


        # The dataloaders
        print(f'-------------------------------------------------------------------')
        print(f'[INFO] Initializing the dataLoaders...')
        self.training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
        self.testing_loader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=False)

        # Initialize TensorBoard writer
        if self.DO_LOG_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=training_options.logging_directory)
            print(f'[INFO] TensorBoard writer initialized at: {training_options.logging_directory}')
            
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #

    def train(self, n_epochs=60, test_frequency=1, logging_frequency=25):
        
        print(f'[INFO] Started training in mode {self.TRAIN_MODE} the model for {n_epochs} epochs on device {self.device}...')
    
        # tracking variables
        iteration = 0
        best_accuracy = 0
        best_reconstruction_loss = 100000000000
        best_avg_kl_div_loss = 100000000000
        SAVED_AT_LEAST_ONCE = False
        
        for epoch in tqdm(range(n_epochs)):
            # set the model to train mode
            self.semantic_covert_model.train()
            
            # iterate over the training dataset
            for (images, labels) in self.training_loader:
                # tracking variables for the current iteration
                total = 0
                correct = 0
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # train the feature extractor model (semantic encoder)
                if self.TRAIN_MODE == TRAINING_MODE.FEATURE_EXTRACTOR:
                    # get the predictions
                    logits = self.semantic_covert_model(images)
                    # compute the loss
                    loss_classification = self.lambda_classification * self.classification_loss(logits, labels)
                    loss = loss_classification
                    # calculate the accuracy
                    _, predicted_class = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted_class == labels).sum().item()
                    accuracy = 100 * correct / total

                # train the feature autoencoder model (JSCC encoder)
                elif self.TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER:
                    # get the predictions
                    if self.training_options.dataset.upper() == 'IID':
                        semantic_information, encoded_information, decoded_semantic_information, noise = self.semantic_covert_model(images)
                    else:
                        semantic_information, encoded_information, decoded_semantic_information, logits, noise = self.semantic_covert_model(images)

                    # generate the noisy codeword to compute the covertness loss when not using the power loss
                    noisy_codeword = encoded_information + noise
                    # compute the loss
                    loss_reconstruction = self.lambda_reconstruction * self.reconstruction_loss(decoded_semantic_information, semantic_information)
                    # classification loss is not used in this mode
                    loss_classification = 0
                    # power constraint loss
                    if self.lambda_power_constraint:                    
                        loss_power_constraint = self.lambda_power_constraint * self.power_constraint_loss(encoded_information)
                    else:
                        loss_power_constraint = 0
                    if self.lambda_covertness_constraint:
                        loss_covertness_constraint = self.lambda_covertness_constraint * self.covertness_constraint_loss(noisy_codeword, noise)
                    else:
                        loss_covertness_constraint = 0
                        
                    # Compute L2 regularization term
                    l2_regularization_loss = 0
                    if self.lambda_l2_regularization:
                        for param in self.semantic_covert_model.parameters():
                            l2_regularization_loss += torch.norm(param, p=2)
                        l2_regularization_loss = l2_regularization_loss * self.lambda_l2_regularization

                    loss = loss_reconstruction + loss_power_constraint + loss_covertness_constraint + l2_regularization_loss
                    
                    avg_kl_div = self.kl_divergence(encoded_information, noise) / images.shape[0] # normalize by the batch size

                    # calculate the accuracy
                    _, predicted_class = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted_class == labels).sum().item()
                    accuracy = 100 * correct / total

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                iteration = iteration + 1
            
            if iteration % logging_frequency == 0:
                
                if self.TRAIN_MODE == TRAINING_MODE.FEATURE_EXTRACTOR:
                    print(f'[INFO-TRAINING] Epoch [{epoch}/{n_epochs}], Accuracy on the training set: {accuracy:.2f}, Classification Loss: {loss_classification.item()}')
                    
                elif self.TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER:
                    print(f'[INFO-TRAINING] Epoch [{epoch}/{n_epochs}], Accuracy on the training set: {accuracy:.2f}, Average KL divergence {avg_kl_div}, Covertness Constraint Loss: {loss_covertness_constraint}, Reconstruction Loss: {loss_reconstruction.item()}, Power Constraint Loss: {loss_power_constraint}')
                    
                if self.DO_LOG_TENSORBOARD:
                    # Log the losses to tensorboard

                    if self.TRAIN_MODE == TRAINING_MODE.FEATURE_EXTRACTOR:
                        self.writer.add_scalar('train/total', loss.item(), global_step=iteration)
                        self.writer.add_scalar('train/classification', loss_classification, global_step=iteration)
                        self.writer.add_scalar('train/accuracy', accuracy, global_step=iteration)

                    elif self.TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER:
                        self.writer.add_scalar('train/total', loss.item(), global_step=iteration)
                        self.writer.add_scalar('train/reconstruction', loss_reconstruction.item(), global_step=iteration)
                        self.writer.add_scalar('train/covertness', loss_covertness_constraint, global_step=iteration)
                        self.writer.add_scalar('train/power', loss_power_constraint, global_step=iteration)
                        self.writer.add_scalar('train/L2_regularization', l2_regularization_loss, global_step=iteration)
                        self.writer.add_scalar('train/accuracy', accuracy, global_step=iteration)
                    
                    # Logging histograms (weights, biases, gradients, etc.)
                    for name, param in self.semantic_covert_model.named_parameters():
                        self.writer.add_histogram(name, param, global_step=iteration)
                        self.writer.add_histogram(name + '/grad', param.grad, global_step=iteration)
            
            # test the model and save the best model
            if epoch % test_frequency == 0:
                if self.TRAIN_MODE == TRAINING_MODE.FEATURE_EXTRACTOR:
                    accuracy_test = self.test(iteration, epoch, n_epochs)

                    if accuracy_test > best_accuracy:
                        best_accuracy = accuracy_test
                        print(f'-------------------------------------------------------------------')
                        print(f'[INFO] Best testing accuracy so far: {best_accuracy:.2f}, saving the model...')
                        self.save_model()
                
                elif self.TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER:
                    accuracy_test, reconstruction_loss_test, avg_kl_div_test = self.test(iteration, epoch, n_epochs)

                    if accuracy_test > best_accuracy and reconstruction_loss_test < best_reconstruction_loss and avg_kl_div_test < self.training_options.target_kl_div:
                        print(f'-------------------------------------------------------------------')
                        print(f'target_kl_div: {self.training_options.target_kl_div}, avg_kl_div_test: {avg_kl_div_test}')
                        best_accuracy = accuracy_test
                        best_reconstruction_loss = reconstruction_loss_test
                        best_avg_kl_div_loss = avg_kl_div_test
                        print(f'-------------------------------------------------------------------')
                        print(f'[INFO] Best testing accuracy so far: {best_accuracy:.2f}, best testing reconstruction loss so far: {best_reconstruction_loss:.4f}, best testing KL divergence so far: {best_avg_kl_div_loss:.6f}, saving the model...') 
                        self.save_model()
                        SAVED_AT_LEAST_ONCE = True
                            

        # save the model
        if self.TRAIN_MODE == TRAINING_MODE.FEATURE_EXTRACTOR:            
            accuracy_final_test = self.test(iteration, epoch, n_epochs)
            if accuracy_final_test > best_accuracy:
                best_accuracy = accuracy_final_test
                print(f'-------------------------------------------------------------------')
                print(f'[INFO] Best testing accuracy so far: {accuracy_final_test:.2f}, saving the model...')
                self.save_model(training_ended=True)
                SAVED_AT_LEAST_ONCE = True
            
            print(f'-------------------------------------------------------------------')
            print(f'[INFO] Training completed! The best testing accuracy: {best_accuracy:.2f}')
            print(f'-------------------------------------------------------------------')

        elif self.TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER:
            reconstruction_loss_final_test = self.test(iteration, epoch, n_epochs)

            if accuracy_test > best_accuracy :
                best_reconstruction_loss = reconstruction_loss_final_test
                print(f'-------------------------------------------------------------------')
                print(f'[INFO] Best testing accuracy so far: {best_accuracy:.2f}, best testing reconstruction loss so far: {best_reconstruction_loss:.4f}, best testing KL divergence so far: {best_avg_kl_div_loss:.6f}, saving the model...') 
                self.save_model(training_ended=True)
                SAVED_AT_LEAST_ONCE = True

            print(f'-------------------------------------------------------------------')
            print(f'[INFO] Training completed! The best testing reconstruction loss: {best_reconstruction_loss:.2f}')
            print(f'-------------------------------------------------------------------')

        if not SAVED_AT_LEAST_ONCE:
            self.save_model(training_ended=True, force_save=True)
            
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
    
    def kl_divergence(self, encoded_information, noise):
        # the output at the warden under the null hypothesis
            z_0 = noise
            # the output at the warden under the alternative hypothesis
            z_1= encoded_information + noise

            z_0 = z_0.detach().cpu().numpy().flatten().squeeze()
            z_1 = z_1.detach().cpu().numpy().flatten().squeeze()
            
            # to avoid nan values
            not_nan_z0 = z_0[~np.isnan(z_0)]
            not_nan_z1 = z_1[~np.isnan(z_1)]
            
            # Compute histograms
            hist_0, bins_0 = np.histogram(not_nan_z0, bins=100)
            hist_1, bins_1 = np.histogram(not_nan_z1, bins=100)
        
            # Filter out zero values
            non_zero_indices_0 = np.where(hist_0 != 0)[0]
            non_zero_indices_1 = np.where(hist_1 != 0)[0]

            # Find common non-zero indices
            common_indices = np.intersect1d(non_zero_indices_0, non_zero_indices_1)

            # Calculate the KL divergence based on non-zero values
            kl_div =  entropy(hist_1[common_indices], hist_0[common_indices])

            return kl_div
        
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
    
    def test(self, iteration, epoch, n_epochs):
        # set the model to evaluation mode
        self.semantic_covert_model.eval()
        # tracking variables
        correct = 0
        total = 0
        loss_reconstruction = 0
        loss_classification = 0
        kl_div = 0
        
        # Evaluate the model on the test set
        with torch.no_grad():
            for (images, labels) in self.testing_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # if we are testing the feature extractor model (semantic encoder)
                if self.TRAIN_MODE == TRAINING_MODE.FEATURE_EXTRACTOR or self.TRAIN_MODE == TRAINING_MODE.TEST_FEATURE_EXTRACTOR:
                    # get the predictions
                    logits = self.semantic_covert_model(images)
                    # compute the loss
                    loss_classification = loss_classification + self.lambda_classification * self.classification_loss(logits, labels)
                    # calculate the accuracy
                    _, predicted_class = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted_class == labels).sum().item()
                    accuracy = 100 * correct / total
                    loss_covertness_constraint = 0
                    loss_power_constraint = 0
                
                # if we are testing the feature autoencoder model (JSCC encoder)
                elif self.TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER:
                    if self.training_options.dataset.upper() == 'IID':
                        semantic_information, encoded_information, decoded_semantic_information, noise = self.semantic_covert_model(images)
                    else:
                        semantic_information, encoded_information, decoded_semantic_information, logits, noise = self.semantic_covert_model(images)
                    # generate the noisy codeword to compute the covertness loss when not using the power loss
                    noisy_codeword = encoded_information + noise
                    # compute the loss               
                    loss_reconstruction = self.lambda_reconstruction * self.reconstruction_loss(decoded_semantic_information, semantic_information)
                    loss_power_constraint = self.lambda_power_constraint * self.power_constraint_loss(encoded_information)
                    if self.lambda_covertness_constraint:
                        loss_covertness_constraint = self.lambda_covertness_constraint * self.covertness_constraint_loss(noisy_codeword, noise)
                    else:
                        loss_covertness_constraint = 0
                        
                    # calculate the accuracy
                    _, predicted_class = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted_class == labels).sum().item()
                    accuracy = 100 * correct / total
                    # compute the KL divergence
                    kl_div = kl_div + self.kl_divergence(encoded_information, noise)
                    
                # if we are testing the end-tp-end model
                elif self.TRAIN_MODE == TRAINING_MODE.TEST:
                    if self.training_options.dataset.upper() == 'IID':
                        semantic_information, encoded_information, decoded_semantic_information, noise = self.semantic_covert_model(images)
                    else:
                        semantic_information, encoded_information, decoded_semantic_information, logits, noise = self.semantic_covert_model(images)

                    # generate the noisy codeword to compute the covertness loss when not using the power loss
                    noisy_codeword = encoded_information + noise
                    # compute the loss
                    loss_reconstruction = self.lambda_reconstruction * self.reconstruction_loss(decoded_semantic_information, semantic_information)
                    loss_classification = self.lambda_classification * self.classification_loss(logits, labels)
                    loss_power_constraint = self.lambda_power_constraint * self.power_constraint_loss(encoded_information)
                    loss_covertness_constraint = self.lambda_covertness_constraint * self.covertness_constraint_loss(noisy_codeword, noise)
                    
                    # calculate the accuracy
                    _, predicted_class = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted_class == labels).sum().item()
                    accuracy = 100 * correct / total

                    kl_div = kl_div + self.kl_divergence(encoded_information, noise)

                    
            avg_loss_classification = loss_classification/len(self.testing_loader)
            avg_loss_reconstruction = loss_reconstruction/len(self.testing_loader)
            avg_loss_covertness_constraint = loss_covertness_constraint/len(self.testing_loader)
            avg_loss_power_constraint = loss_power_constraint/len(self.testing_loader)
            avg_kl_div = kl_div/len(self.testing_loader)
            
            if self.TRAIN_MODE == TRAINING_MODE.FEATURE_EXTRACTOR:
                print(f'[INFO-TEST] Epoch [{epoch}/{n_epochs}], Accuracy on the test set: {accuracy:.2f}, Average Classification Loss: {avg_loss_classification}')
                
            elif self.TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER:
                print(f'[INFO-TEST] Epoch [{epoch}/{n_epochs}], Accuracy on the test set: {accuracy:.2f}, Average KL divergence {avg_kl_div}, Covertness Constraint Loss: {loss_covertness_constraint}, Average Reconstruction Loss: {avg_loss_reconstruction}, Average Power Constraint Loss: {loss_power_constraint}')
                
        # log to tensorboard  
        if self.DO_LOG_TENSORBOARD:

            if self.TRAIN_MODE == TRAINING_MODE.FEATURE_EXTRACTOR:
                self.writer.add_scalar('test/accuracy', accuracy, global_step=iteration)
                self.writer.add_scalar('test/avg_classification', avg_loss_classification, global_step=iteration)
                return accuracy

            elif self.TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER:
                self.writer.add_scalar('test/accuracy', accuracy, global_step=iteration)
                self.writer.add_scalar('test/avg_reconstruction', avg_loss_reconstruction, global_step=iteration)
                self.writer.add_scalar('test/avg_covertness', avg_loss_covertness_constraint, global_step=iteration)
                self.writer.add_scalar('test/avg_power_constraint', avg_loss_power_constraint, global_step=iteration)
                self.writer.add_scalar('test/avg_kl_div', avg_kl_div, global_step=iteration)
                return accuracy, avg_loss_reconstruction, avg_kl_div
            else:
                self.writer.add_scalar('test/accuracy', accuracy, global_step=iteration)
                self.writer.add_scalar('test/avg_reconstruction', avg_loss_reconstruction, global_step=iteration)
                self.writer.add_scalar('test/avg_classification', avg_loss_classification, global_step=iteration)
                self.writer.add_scalar('test/avg_covertness', avg_loss_covertness_constraint, global_step=iteration)
                self.writer.add_scalar('test/avg_power_constraint', avg_loss_power_constraint, global_step=iteration)
                self.writer.add_scalar('test/avg_kl_div', avg_kl_div, global_step=iteration)

                return accuracy, avg_loss_reconstruction, avg_kl_div
            
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #
    
    def save_model(self, training_ended=False, force_save=False):
        
        if self.TRAIN_MODE == TRAINING_MODE.FEATURE_EXTRACTOR:
        
            torch.save(self.semantic_covert_model.semantic_encoder.state_dict(), self.SEMANTIC_ENCODER_CKPT_SAVING_PATH)
            torch.save(self.semantic_covert_model.classifier.state_dict(), self.CLASSIFIER_CKPT_SAVING_PATH)
            if training_ended:
                print(f'-------------------------------------------------------------------')
                print(f'[INFO] Semantic Encoder saved at: {self.SEMANTIC_ENCODER_CKPT_SAVING_PATH}')
                print(f'[INFO] Classifier saved at: {self.CLASSIFIER_CKPT_SAVING_PATH}')
                print(f'-------------------------------------------------------------------')
            
        elif self.TRAIN_MODE == TRAINING_MODE.AUTO_ENCODER:

            if force_save:
                print(f'-------------------------------------------------------------------')
                print(f'[INFO] Forced saving the model...')
                print(f'-------------------------------------------------------------------')
                # The path to the JSCC encoder checkpoint             
                JSCC_ENCODER_CKPT_SAVING_PATH = f'{self.training_options.checkpoint_directory}/jscc_encoder_forced_last.pth'
                # The path to the semantic decoder checkpoint         
                SEMANTIC_DECODER_CKPT_SAVING_PATH = f'{self.training_options.checkpoint_directory}/semantic_decoder_forced_last.pth'
                torch.save(self.semantic_covert_model.jscc_encoder.state_dict(), JSCC_ENCODER_CKPT_SAVING_PATH)
                torch.save(self.semantic_covert_model.semantic_decoder.state_dict(), SEMANTIC_DECODER_CKPT_SAVING_PATH)

                print(f'-------------------------------------------------------------------')
                print(f'[INFO] JSCC Encoder saved at: {JSCC_ENCODER_CKPT_SAVING_PATH}')
                print(f'[INFO] Semantic Decoder saved at: {SEMANTIC_DECODER_CKPT_SAVING_PATH}')
                print(f'-------------------------------------------------------------------')
            else:
                torch.save(self.semantic_covert_model.jscc_encoder.state_dict(), self.JSCC_ENCODER_CKPT_SAVING_PATH)
                torch.save(self.semantic_covert_model.semantic_decoder.state_dict(), self.SEMANTIC_DECODER_CKPT_SAVING_PATH)

                if training_ended:
                    print(f'-------------------------------------------------------------------')
                    print(f'[INFO] JSCC Encoder saved at: {self.JSCC_ENCODER_CKPT_SAVING_PATH}')
                    print(f'[INFO] Semantic Decoder saved at: {self.SEMANTIC_DECODER_CKPT_SAVING_PATH}')
                    print(f'-------------------------------------------------------------------')
            
    # --------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------------------------------------------------------------- #