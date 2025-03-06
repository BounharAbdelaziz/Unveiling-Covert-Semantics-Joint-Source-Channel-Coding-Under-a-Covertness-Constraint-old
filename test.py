import numpy as np
import matplotlib.pyplot as plt

import torch

from model.trainer import Trainer
from model.trainer import TRAINING_MODE

import numpy as np
from options.base_options import Options

from scipy.stats import entropy

from tqdm import tqdm
from scipy.optimize import curve_fit
import os


def run_test(trainer: Trainer, MAX_TEST_SAMPLES: int= 1000, batch_size: int = 100, SNR: float = 0.0, model: str='covert', PLOT_PATH:str = './plots/acc=f(n)_best_only_more_n/'):
    
    # set the model to evaluation mode
    trainer.semantic_covert_model.eval()
    
    # Evaluate the model on the test set
    with torch.no_grad():
        print(f'[INFO] Testing the model at SNR = {SNR} dB...')
        # tracking variables
        correct = []
        kl_divs = []
        cpt = 0
        loss_reconstruction = 0
        losses_reconstruction = []
        
        # print(f'[INFO] Testing the model : {trainer.semantic_covert_model}')
        for images, labels in tqdm(trainer.testing_loader):

            images = images.to(trainer.device)
            labels = labels.to(trainer.device)
            
            if testing_options.dataset.upper() == 'IID':
                _, encoded_information, decoded_semantic_information, noise = trainer.semantic_covert_model(images)
                loss_reconstruction += trainer.reconstruction_loss(images, decoded_semantic_information).item()
            else:
                semantic_information, encoded_information, decoded_semantic_information, logits, noise = trainer.semantic_covert_model(images)
                
                loss_reconstruction = trainer.reconstruction_loss(semantic_information, decoded_semantic_information).item()
                losses_reconstruction.append(loss_reconstruction)
                # print(f'loss_reconstruction: {loss_reconstruction}')
                # calculate the accuracy
                _, predicted_class = torch.max(logits.data, 1)
                correct.append( (predicted_class == labels).sum().item())
                
            # the output at the warden under the null hypothesis
            z_0 = noise
            # the output at the warden under the alternative hypothesis
            z_1= encoded_information + noise

            z_0 = z_0.cpu().numpy().flatten().squeeze()
            z_1 = z_1.cpu().numpy().flatten().squeeze()
            
            # Compute histograms
            hist_0, bins_0 = np.histogram(z_0, bins=100)
            hist_1, bins_1 = np.histogram(z_1, bins=100)


            plt.figure(figsize=(10, 5))
            plt.plot(bins_0[:-1], hist_0, label='Distribution under $\mathcal{H}=0$')
            plt.plot(bins_1[:-1], hist_1, label='Distribution under $\mathcal{H}=1$')
            plt.title('Output distribution under the two hypotheses covert model')
            plt.xlabel('Bin')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
            plt.savefig(f'{PLOT_PATH}/histogram_{model}.png')
            plt.close()
        
            # Filter out zero values
            non_zero_indices_0 = np.where(hist_0 != 0)[0]
            non_zero_indices_1 = np.where(hist_1 != 0)[0]

            # Find common non-zero indices
            common_indices = np.intersect1d(non_zero_indices_0, non_zero_indices_1)

            # Calculate the KL divergence based on non-zero values
            kl_div =  entropy(hist_1[common_indices], hist_0[common_indices])
            kl_divs.append(kl_div)
            
            print('-----------------------------')
            print(f'real kl_div: {kl_div}')
            
            cpt += batch_size
            if cpt == MAX_TEST_SAMPLES:
                break
            
        accuracy = 100 * sum(correct) / MAX_TEST_SAMPLES
        avg_loss_reconstruction = np.average(losses_reconstruction)
        avg_kl_div = np.average(kl_divs)
        
        print(f'[INFO] avg_kl_div: {avg_kl_div}')
    
    return accuracy, avg_loss_reconstruction, avg_kl_div
    

if __name__ == "__main__":

    PLOT_PATH = './plots/acc=f(n)_best_only_more_n/'

    if not os.path.exists(PLOT_PATH):
        os.makedirs(PLOT_PATH)
        print(f'[INFO] Created directory: {PLOT_PATH}')
    
    PLOT_FIT = False
    WITH_SK = False
    sk_path = 'with-sk' if WITH_SK else 'without-sk' 
    plt.figure(figsize=(15, 6))
    n_steps = 1 #4 # 5 # 20
    min_SNR = 1 #5 #1
    max_SNR = 1 #5 #1 #12
    SNR_levels = np.linspace(min_SNR, max_SNR, n_steps)
    

    print(f'[INFO] Testing with SNR_levels: {SNR_levels}')
    
    TRAIN_MODE = TRAINING_MODE.TEST

    # the options
    testing_options = Options(TRAIN_MODE).parse()

    # the maximum number of test samples
    MAX_TEST_SAMPLES = testing_options.MAX_TEST_SAMPLES
    
    # a dictionary to set the values of n and k for the covert model to test   
    if testing_options.dataset.upper() == 'MNIST':
      
        # k_n_sqrt_dict = {
        #     "512": [1, 3, 4, 6, 7],
        #     "768": [1, 2, 9, 12],
        #     "1024": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        #     "2048": [2, 4, 5, 8, 10, 11, 12, 14],
        #     "4096": [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        # }
        
        # k_n_dict = {
        #     "512": [102, 409, 512],
        #     "768": [153, 614, 768],
        #     "1024": [204, 819, 1024],
        #     "2048": [409, 1638, 2048],
        #     "4096": [819, 3276, 4096],
        # }
        
        k_n_sqrt_dict = {
            "512": [6],
            "2048": [11],
        }
        
        k_n_dict = {
            "512": [102, 409],
            "2048": [409, 1638],
        }

    if testing_options.do_quantize:
        is_qantized_enc = 'quantized-enc'
    else:
        is_qantized_enc = 'not-quantized-enc'
    
    if testing_options.do_quantize_U:
        is_qantized_sem = 'quantized-sem'
    else:
        is_qantized_sem = 'not-quantized-sem'

    if testing_options.do_quantize_U_hat:
        is_qantized_dec = 'quantized-dec'
    else:
        is_qantized_dec = 'not-quantized-dec'
        
    # # -------------------------------------------- Test the Non-Covert model (trained with k=O(n))------------------------------------------------ #
    model_type = 'non-covert'
    # LOAD_FORCED = True
    LOAD_FORCED = False

    # Initialize lists to store handles and labels
    handles, labels = [], []
    
    for SNR_test in SNR_levels:
        
        mse_values_dict = {}  # Initialize dictionary to store MSE values for each n
        
        max_accuracy_for_each_n = []
        best_reconstruction_for_each_n = []
        best_kl_div_for_each_n = []
        
        for n, k_list in k_n_dict.items():

            accuracy_for_each_k = []
            reconstruction_for_each_k = []
            kl_div_for_each_k = []

            for k in k_list:
                print(f'[INFO] Testing the covert model O(n) at SNR = {SNR_test} dB...')
                
                print(f'[INFO] Testing the covert model with n = {n} and k = {k}')

                testing_options.k = int(k)
                testing_options.blocklength = int(n)
                testing_options.output_dim_jscc_encoder = int(n)
                testing_options.hidden_dim_jscc_encoder = int(n)
                testing_options.input_dim_jscc_decoder = int(n)
                testing_options.hidden_dim_jscc_decoder = int(n)
                
                testing_options.train_covert_model = False
                testing_options.test_with_joint_model = False
                # testing_options.test_with_joint_model = True

                # testing_options.checkpoint_directory = f'./checkpoints/{testing_options.dataset.upper()}/n/{testing_options.blocklength}/{model_type}/{is_qantized_sem}/{is_qantized_enc}/C/{testing_options.C}/A/{testing_options.A}/k/{testing_options.k}/SNR/{str(testing_options.SNR)}_db/'
                testing_options.checkpoint_directory = f'./checkpoints/{sk_path}/{testing_options.dataset.upper()}/n/{testing_options.blocklength}/{model_type}/{is_qantized_sem}/{is_qantized_enc}/{is_qantized_dec}/C/{testing_options.C}/A/{testing_options.A}/k/{testing_options.k}/SNR/{str(testing_options.SNR)}_db/'
                # max power constraint
                maximum_power = testing_options.A * np.sqrt(testing_options.blocklength) # * n
                print(f'[INFO] Checkpoint directory: {testing_options.checkpoint_directory}')
                
                # Create the trainer     
                trainer = Trainer( testing_options,                     # the training options
                                    maximum_power,                      # the maximum power constraint
                                    TRAIN_MODE,                         # Which model to train Train the model jointly or separately
                                    LOAD_FORCED=LOAD_FORCED,
                        )
                
                # set the SNR
                trainer.semantic_covert_model.channel.set_snr(SNR_test)

                accuracy, avg_loss_reconstruction, avg_kl_div = run_test(trainer, MAX_TEST_SAMPLES, testing_options.batch_size, SNR_test, model=f'non_covert_{testing_options.k}')
                accuracy_for_each_k.append(accuracy)
                reconstruction_for_each_k.append(avg_loss_reconstruction)
                kl_div_for_each_k.append(avg_kl_div)

                if testing_options.dataset.upper() == 'IID':
                    print(f'for k = {k} and SNR = {SNR_test}, the average loss reconstruction is {avg_loss_reconstruction} and the average KL divergence is {avg_kl_div}')
                else:
                    print(f'for k = {k} and SNR = {SNR_test}, the accuracy is {accuracy} and the average KL divergence is {avg_kl_div}')

            
            index_max_accuracy_k = np.argmax(accuracy_for_each_k)
            max_accuracy_k = accuracy_for_each_k[index_max_accuracy_k]
            kl_div_k_of_max = kl_div_for_each_k[index_max_accuracy_k]
            best_k_value = k_list[index_max_accuracy_k]
            
            max_accuracy_for_each_n.append(max_accuracy_k)
            best_kl_div_for_each_n.append(kl_div_k_of_max)
            
            # log the results
            print(f'for k = {k} and SNR = {SNR_test}')
            print(f'kl_div_for_each_k: {kl_div_for_each_k}')
            if testing_options.dataset.upper() == 'IID':
                print(f'reconstruction_for_each_k: {reconstruction_for_each_k}')
            else:
                print(f'accuracy_for_each_k: {accuracy_for_each_k}')
            print(f'best_k_value: {best_k_value}')
            print('-----------------------------------------------------------------------------------------------------------------')
            
            with open(f'{PLOT_PATH}/{testing_options.dataset}_nsamples_{testing_options.MAX_TEST_SAMPLES}_min_SNR_{min_SNR}_max_SNR_{max_SNR}_many_k_epsilon_n_test_{testing_options.epsilon_n}_SNRtrain_{testing_options.SNR}.txt', 'a') as f:
                if testing_options.dataset.upper() == 'IID':
                    f.write(f'for n = {n} and SNR = {SNR_test}\n')
                    f.write(f'reconstruction_for_each_k: {reconstruction_for_each_k}\n')
                else:
                    f.write(f'for n = {n} and SNR = {SNR_test}\n')
                    f.write(f'accuracy_for_each_k: {accuracy_for_each_k}\n')
                    f.write(f'best_k_value: {best_k_value}\n')

                f.write(f'kl_div_for_each_k: {kl_div_for_each_k}\n')
                f.write('-----------------------------------------------------------------------------------------------------------------\n')
        
        
        with open(f'{PLOT_PATH}/{testing_options.dataset}_nsamples_{testing_options.MAX_TEST_SAMPLES}_min_SNR_{min_SNR}_max_SNR_{max_SNR}_many_k_epsilon_n_test_{testing_options.epsilon_n}_SNRtrain_{testing_options.SNR}.txt', 'a') as f:
            if testing_options.dataset.upper() == 'IID':
                f.write(f'for n = {n} and SNR = {SNR_test}\n')
                f.write(f'best_reconstruction_for_each_n: {best_reconstruction_for_each_n}\n')
            else:
                f.write(f'for SNR = {SNR_test}\n')
                f.write(f'max_accuracy_for_each_n: {max_accuracy_for_each_n}\n')
                f.write(f'best_k_value: {best_k_value}\n')
            f.write(f'best_kl_div_for_each_n: {best_kl_div_for_each_n}\n')
            f.write('-----------------------------------------------------------------------------------------------------------------\n')
        
        if testing_options.dataset.upper() == 'IID':
            k_list = [float(n) for n in k_n_dict.keys()]
            if len(SNR_levels) > 1:
                # Subplot for MSE=f(k)
                plt.subplot(2, 1, 1)
                plt.plot(k_list, reconstruction_for_each_k, linestyle='-', label=f'n = {n}, SNR= {SNR_test}', color='red')
                plt.text(k_list[-1], reconstruction_for_each_k[-1], f'n = {n}', verticalalignment='center', fontsize=8)
                # Subplot for KL=f(k)
                plt.subplot(2, 1, 2)
                plt.plot(k_list, kl_div_for_each_k, label=f'n = {n}, SNR= {SNR_test}', color='red')
                plt.text(k_list[-1], kl_div_for_each_k[-1], f'n = {n}', verticalalignment='center', fontsize=8)

            else:
                # Subplot for MSE=f(k)
                plt.subplot(2, 1, 1)
                plt.plot(k_list, reconstruction_for_each_k, linestyle='-', label=f'n = {n}', color='red')
                plt.text(k_list[-1], reconstruction_for_each_k[-1], f'n = {n}', verticalalignment='center', fontsize=8)
                # Subplot for KL=f(k)
                plt.subplot(2, 1, 2)
                plt.plot(k_list, kl_div_for_each_k, linestyle='-', label=f'n = {n}', color='red')
                plt.text(k_list[-1], kl_div_for_each_k[-1], f'n = {n}', verticalalignment='center', fontsize=8)

        else:
            n_list = [float(n) for n in k_n_dict.keys()]
            
            if len(SNR_levels) > 1:
                # Subplot for accuracy=f(n)
                plt.subplot(2, 1, 1)  
                plt.plot(n_list, max_accuracy_for_each_n, linestyle='-', label=f'k = O(n) / non-covert, SNR= {SNR_test}', color='red')
                # Subplot for KL=f(k)
                plt.subplot(2, 1, 2)
                plt.plot(n_list, best_kl_div_for_each_n, linestyle='-', label=f'k = O(n) / non-covert, SNR= {SNR_test}', color='red')

            else:
                # Subplot for accuracy=f(n)
                plt.subplot(2, 1, 1)  
                plt.plot(n_list, max_accuracy_for_each_n, linestyle='-', label=f'k = O(n) / non-covert', color='red')
                # Subplot for KL=f(k)
                plt.subplot(2, 1, 2)
                plt.plot(n_list, best_kl_div_for_each_n, linestyle='-', label=f'k = O(n) / non-covert', color='red')
    
    # -------------------------------------------- Test the Covert model (trained with k=O(n))------------------------------------------------ #
    model_type = 'non-covert-KL-training'

    # Initialize lists to store handles and labels
    handles, labels = [], []
    
    for SNR_test in SNR_levels:
        
        mse_values_dict = {}  # Initialize dictionary to store MSE values for each n
        
        max_accuracy_for_each_n = []
        best_reconstruction_for_each_n = []
        best_kl_div_for_each_n = []
        
        for n, k_list in k_n_dict.items():
            # if n != '512':
            #     LOAD_FORCED = False
            # else:

            accuracy_for_each_k = []
            reconstruction_for_each_k = []
            kl_div_for_each_k = []

            for k in k_list:
                print(f'[INFO] Testing the covert model O(n) at SNR = {SNR_test} dB...')
                
                print(f'[INFO] Testing the covert model with n = {n} and k = {k}')

                testing_options.k = int(k)
                testing_options.blocklength = int(n)
                testing_options.output_dim_jscc_encoder = int(n)
                testing_options.hidden_dim_jscc_encoder = int(n)
                testing_options.input_dim_jscc_decoder = int(n)
                testing_options.hidden_dim_jscc_decoder = int(n)
                
                testing_options.train_covert_model = True
                testing_options.test_with_joint_model = False
                # testing_options.test_with_joint_model = True

                # testing_options.checkpoint_directory = f'./checkpoints/{testing_options.dataset.upper()}/n/{testing_options.blocklength}/{model_type}/{is_qantized_sem}/{is_qantized_enc}/C/{testing_options.C}/A/{testing_options.A}/k/{testing_options.k}/SNR/{str(testing_options.SNR)}_db/'
                testing_options.checkpoint_directory = f'./checkpoints/{sk_path}/{testing_options.dataset.upper()}/n/{testing_options.blocklength}/{model_type}/{is_qantized_sem}/{is_qantized_enc}/{is_qantized_dec}/C/{testing_options.C}/A/{testing_options.A}/k/{testing_options.k}/SNR/{str(testing_options.SNR)}_db/'
                # max power constraint
                maximum_power = testing_options.A * np.sqrt(testing_options.blocklength) # * n
                print(f'[INFO] Checkpoint directory: {testing_options.checkpoint_directory}')
                
                # Create the trainer     
                trainer = Trainer( testing_options,                     # the training options
                                    maximum_power,                      # the maximum power constraint
                                    TRAIN_MODE,                         # Which model to train Train the model jointly or separately
                                    LOAD_FORCED=LOAD_FORCED,
                        )
                
                # set the SNR
                trainer.semantic_covert_model.channel.set_snr(SNR_test)

                accuracy, avg_loss_reconstruction, avg_kl_div = run_test(trainer, MAX_TEST_SAMPLES, testing_options.batch_size, SNR_test, model=f'linear_covert_{testing_options.k}')
                accuracy_for_each_k.append(accuracy)
                reconstruction_for_each_k.append(avg_loss_reconstruction)
                kl_div_for_each_k.append(avg_kl_div)

                if testing_options.dataset.upper() == 'IID':
                    print(f'for k = {k} and SNR = {SNR_test}, the average loss reconstruction is {avg_loss_reconstruction} and the average KL divergence is {avg_kl_div}')
                else:
                    print(f'for k = {k} and SNR = {SNR_test}, the accuracy is {accuracy} and the average KL divergence is {avg_kl_div}')

            
            index_max_accuracy_k = np.argmax(accuracy_for_each_k)
            max_accuracy_k = accuracy_for_each_k[index_max_accuracy_k]
            kl_div_k_of_max = kl_div_for_each_k[index_max_accuracy_k]
            best_k_value = k_list[index_max_accuracy_k]
            
            max_accuracy_for_each_n.append(max_accuracy_k)
            best_kl_div_for_each_n.append(kl_div_k_of_max)
            
            # log the results
            print(f'for k = {k} and SNR = {SNR_test}')
            print(f'kl_div_for_each_k: {kl_div_for_each_k}')
            if testing_options.dataset.upper() == 'IID':
                print(f'reconstruction_for_each_k: {reconstruction_for_each_k}')
            else:
                print(f'accuracy_for_each_k: {accuracy_for_each_k}')
            print(f'best_k_value: {best_k_value}')
            print('-----------------------------------------------------------------------------------------------------------------')
            
            with open(f'{PLOT_PATH}/{testing_options.dataset}_nsamples_{testing_options.MAX_TEST_SAMPLES}_min_SNR_{min_SNR}_max_SNR_{max_SNR}_many_k_epsilon_n_test_{testing_options.epsilon_n}_SNRtrain_{testing_options.SNR}.txt', 'a') as f:
                if testing_options.dataset.upper() == 'IID':
                    f.write(f'for n = {n} and SNR = {SNR_test}\n')
                    f.write(f'reconstruction_for_each_k: {reconstruction_for_each_k}\n')
                else:
                    f.write(f'for n = {n} and SNR = {SNR_test}\n')
                    f.write(f'accuracy_for_each_k: {accuracy_for_each_k}\n')
                    f.write(f'best_k_value: {best_k_value}\n')

                f.write(f'kl_div_for_each_k: {kl_div_for_each_k}\n')
                f.write('-----------------------------------------------------------------------------------------------------------------\n')
        
        
        with open(f'{PLOT_PATH}/{testing_options.dataset}_nsamples_{testing_options.MAX_TEST_SAMPLES}_min_SNR_{min_SNR}_max_SNR_{max_SNR}_many_k_epsilon_n_test_{testing_options.epsilon_n}_SNRtrain_{testing_options.SNR}.txt', 'a') as f:
            if testing_options.dataset.upper() == 'IID':
                f.write(f'for n = {n} and SNR = {SNR_test}\n')
                f.write(f'best_reconstruction_for_each_n: {best_reconstruction_for_each_n}\n')
            else:
                f.write(f'for SNR = {SNR_test}\n')
                f.write(f'max_accuracy_for_each_n: {max_accuracy_for_each_n}\n')
                f.write(f'best_k_value: {best_k_value}\n')
            f.write(f'best_kl_div_for_each_n: {best_kl_div_for_each_n}\n')
            f.write('-----------------------------------------------------------------------------------------------------------------\n')
        
        if testing_options.dataset.upper() == 'IID':
            k_list = [float(n) for n in k_n_dict.keys()]
            if len(SNR_levels) > 1:
                # Subplot for MSE=f(k)
                plt.subplot(2, 1, 1)
                plt.plot(k_list, reconstruction_for_each_k, linestyle='-', label=f'n = {n}, SNR= {SNR_test}', color='black')
                plt.text(k_list[-1], reconstruction_for_each_k[-1], f'n = {n}', verticalalignment='center', fontsize=8)
                # Subplot for KL=f(k)
                plt.subplot(2, 1, 2)
                plt.plot(k_list, kl_div_for_each_k, label=f'n = {n}, SNR= {SNR_test}', color='black')
                plt.text(k_list[-1], kl_div_for_each_k[-1], f'n = {n}', verticalalignment='center', fontsize=8)

            else:
                # Subplot for MSE=f(k)
                plt.subplot(2, 1, 1)
                plt.plot(k_list, reconstruction_for_each_k, linestyle='-', label=f'n = {n}', color='black')
                plt.text(k_list[-1], reconstruction_for_each_k[-1], f'n = {n}', verticalalignment='center', fontsize=8)
                # Subplot for KL=f(k)
                plt.subplot(2, 1, 2)
                plt.plot(k_list, kl_div_for_each_k, linestyle='-', label=f'n = {n}', color='black')
                plt.text(k_list[-1], kl_div_for_each_k[-1], f'n = {n}', verticalalignment='center', fontsize=8)

        else:
            n_list = [float(n) for n in k_n_dict.keys()]
            
            if len(SNR_levels) > 1:
                # Subplot for accuracy=f(n)
                plt.subplot(2, 1, 1)  
                plt.plot(n_list, max_accuracy_for_each_n, linestyle='-', label=f'k = O(n), SNR= {SNR_test}', color='black')
                # Subplot for KL=f(k)
                plt.subplot(2, 1, 2)
                plt.plot(n_list, best_kl_div_for_each_n, linestyle='-', label=f'k = O(n), SNR= {SNR_test}', color='black')

            else:
                # Subplot for accuracy=f(n)
                plt.subplot(2, 1, 1)  
                plt.plot(n_list, max_accuracy_for_each_n, linestyle='-', label=f'k = O(n)', color='black')
                # Subplot for KL=f(k)
                plt.subplot(2, 1, 2)
                plt.plot(n_list, best_kl_div_for_each_n, linestyle='-', label=f'k = O(n)', color='black')

    # -------------------------------------------- Test the Covert model (trained with k=O(sqrt(n*epsilon_n)))------------------------------------------------ #
    model_type = 'covert'
    
    for SNR_test in SNR_levels:
        
        mse_values_dict = {}  # Initialize dictionary to store MSE values for each n
        accuracy_values_dict = {}  # Initialize dictionary to store accuracy values for each n
        max_accuracy_for_each_n = []
        best_reconstruction_for_each_n = []
        best_kl_div_for_each_n = []
        
        for n, k_list in k_n_sqrt_dict.items():
            accuracy_for_each_k = []
            reconstruction_for_each_k = []
            kl_div_for_each_k = []

            for k in k_list:
                
                print(f'[INFO] Testing the covert model O(sqrt(n*epsilon_n)) at SNR = {SNR_test} dB...')
                
                print(f'[INFO] Testing the covert model with n = {n} and k = {k}')

                testing_options.k = int(k)
                testing_options.blocklength = int(n)
                testing_options.output_dim_jscc_encoder = int(n)
                testing_options.hidden_dim_jscc_encoder = int(n)
                testing_options.input_dim_jscc_decoder = int(n)
                testing_options.hidden_dim_jscc_decoder = int(n)
                
                testing_options.train_covert_model = True
                testing_options.test_with_joint_model = False
                # testing_options.test_with_joint_model = True

                # testing_options.checkpoint_directory = f'./checkpoints/{testing_options.dataset.upper()}/n/{testing_options.blocklength}/{model_type}/{is_qantized_sem}/{is_qantized_enc}/C/{testing_options.C}/A/{testing_options.A}/k/{testing_options.k}/SNR/{str(testing_options.SNR)}_db/'
                testing_options.checkpoint_directory = f'./checkpoints/{sk_path}/{testing_options.dataset.upper()}/n/{testing_options.blocklength}/{model_type}/{is_qantized_sem}/{is_qantized_enc}/{is_qantized_dec}/C/{testing_options.C}/A/{testing_options.A}/k/{testing_options.k}/SNR/{str(testing_options.SNR)}_db/'
              
                # max power constraint
                maximum_power = testing_options.A * np.sqrt(testing_options.blocklength) # * n
                print(f'[INFO] Checkpoint directory: {testing_options.checkpoint_directory}')
                
                # Create the trainer     
                trainer = Trainer( testing_options,                     # the training options
                                    maximum_power,                      # the maximum power constraint
                                    TRAIN_MODE,                         # Which model to train Train the model jointly or separately
                        )
                
                # set the SNR
                trainer.semantic_covert_model.channel.set_snr(SNR_test)

                accuracy, avg_loss_reconstruction, avg_kl_div = run_test(trainer, MAX_TEST_SAMPLES, testing_options.batch_size, SNR_test, model=f'covert_{testing_options.k}')
                accuracy_for_each_k.append(accuracy)
                reconstruction_for_each_k.append(avg_loss_reconstruction)
                kl_div_for_each_k.append(avg_kl_div)

                if testing_options.dataset.upper() == 'IID':
                    print(f'for k = {k} and SNR = {SNR_test}, the average loss reconstruction is {avg_loss_reconstruction} and the average KL divergence is {avg_kl_div}')
                else:
                    print(f'for k = {k} and SNR = {SNR_test}, the accuracy is {accuracy} and the average KL divergence is {avg_kl_div}')

            
            index_max_accuracy_k = np.argmax(accuracy_for_each_k)
            max_accuracy_k = accuracy_for_each_k[index_max_accuracy_k]
            kl_div_k_of_max = kl_div_for_each_k[index_max_accuracy_k]
            best_k_value = k_list[index_max_accuracy_k]
            
            max_accuracy_for_each_n.append(max_accuracy_k)
            best_kl_div_for_each_n.append(kl_div_k_of_max)
            
            # log the results
            print(f'for k = {k} and SNR = {SNR_test}')
            print(f'kl_div_for_each_k: {kl_div_for_each_k}')
            if testing_options.dataset.upper() == 'IID':
                print(f'reconstruction_for_each_k: {reconstruction_for_each_k}')
            else:
                print(f'accuracy_for_each_k: {accuracy_for_each_k}')
            print(f'best_k_value: {best_k_value}')
            print('-----------------------------------------------------------------------------------------------------------------')
            
            with open(f'{PLOT_PATH}/{testing_options.dataset}_nsamples_{testing_options.MAX_TEST_SAMPLES}_min_SNR_{min_SNR}_max_SNR_{max_SNR}_many_k_epsilon_n_test_{testing_options.epsilon_n}_SNRtrain_{testing_options.SNR}.txt', 'a') as f:
                if testing_options.dataset.upper() == 'IID':
                    f.write(f'for n = {n} and SNR = {SNR_test}\n')
                    f.write(f'reconstruction_for_each_k: {reconstruction_for_each_k}\n')
                else:
                    f.write(f'for n = {n} and SNR = {SNR_test}\n')
                    f.write(f'accuracy_for_each_k: {accuracy_for_each_k}\n')
                f.write(f'kl_div_for_each_k: {kl_div_for_each_k}\n')
                f.write(f': {best_k_value}\n')
                f.write('-----------------------------------------------------------------------------------------------------------------\n')
        
        
        with open(f'{PLOT_PATH}/{testing_options.dataset}_nsamples_{testing_options.MAX_TEST_SAMPLES}_min_SNR_{min_SNR}_max_SNR_{max_SNR}_many_k_epsilon_n_test_{testing_options.epsilon_n}_SNRtrain_{testing_options.SNR}.txt', 'a') as f:
            if testing_options.dataset.upper() == 'IID':
                f.write(f'for n = {n} and SNR = {SNR_test}\n')
                f.write(f'best_reconstruction_for_each_n: {best_reconstruction_for_each_n}\n')
            else:
                f.write(f'for SNR = {SNR_test}\n')
                f.write(f'max_accuracy_for_each_n: {max_accuracy_for_each_n}\n')
            f.write(f'best_k_value: {best_k_value}\n')
            f.write(f'best_kl_div_for_each_n: {best_kl_div_for_each_n}\n')
            f.write('-----------------------------------------------------------------------------------------------------------------\n')
        
        if testing_options.dataset.upper() == 'IID':
            k_list = [float(n) for n in k_n_dict.keys()]
            if len(SNR_levels) > 1:
                # Subplot for MSE=f(k)
                plt.subplot(2, 1, 1)
                plt.plot(k_list, reconstruction_for_each_k, linestyle='--', label=f'n = {n}, SNR= {SNR_test}', color='green')
                plt.text(k_list[-1], reconstruction_for_each_k[-1], f'n = {n}', verticalalignment='center', fontsize=8)
                # Subplot for KL=f(k)
                plt.subplot(2, 1, 2)
                plt.plot(k_list, kl_div_for_each_k, linestyle='--', label=f'n = {n}, SNR= {SNR_test}', color='green')
                plt.text(k_list[-1], kl_div_for_each_k[-1], f'n = {n}', verticalalignment='center', fontsize=8)

            else:
                # Subplot for MSE=f(k)
                plt.subplot(2, 1, 1)
                plt.plot(k_list, reconstruction_for_each_k, linestyle='--', label=f'n = {n}', color='green')
                plt.text(k_list[-1], reconstruction_for_each_k[-1], f'n = {n}', verticalalignment='center', fontsize=8)
                # Subplot for KL=f(k)
                plt.subplot(2, 1, 2)
                plt.plot(k_list, kl_div_for_each_k, linestyle='--', label=f'n = {n}', color='green')
                plt.text(k_list[-1], kl_div_for_each_k[-1], f'n = {n}', verticalalignment='center', fontsize=8)

        else:
            n_list = [float(n) for n in k_n_dict.keys()]
            
            if len(SNR_levels) > 1:
                # Subplot for accuracy=f(n)
                plt.subplot(2, 1, 1)  
                plt.plot(n_list, max_accuracy_for_each_n, linestyle='--', label=f'k = O(sqrt(n*delta)), SNR= {SNR_test}', color='green')
                # Subplot for KL=f(k)
                plt.subplot(2, 1, 2)
                plt.plot(n_list, best_kl_div_for_each_n, linestyle='--', label=f'k = O(sqrt(n*delta)), SNR= {SNR_test}', color='green')

            else:
                # Subplot for accuracy=f(n)
                plt.subplot(2, 1, 1)  
                plt.plot(n_list, max_accuracy_for_each_n, linestyle='--', label=f'k = O(sqrt(n*delta))', color='green')
                # Subplot for KL=f(k)
                plt.subplot(2, 1, 2)
                plt.plot(n_list, best_kl_div_for_each_n, linestyle='--', label=f'k = O(sqrt(n*delta))', color='green')

    # Set spacing between subplots
    plt.subplots_adjust(hspace=0.5)
    
    # Get handles and labels for all plots
    handles, labels = plt.gca().get_legend_handles_labels()

    plt.subplot(2, 1, 1)  # Subplot for MSE
    if testing_options.dataset.upper() == 'IID':
        plt.xlabel('k')
        plt.ylabel('Log Reconstruction Loss')
        plt.yscale('log')  # Set y-axis to log scale
        plt.title('Reconstruction Loss vs. k for Different n')
        
        plt.subplot(2, 1, 2)  # Subplot for accuracy
        plt.xlabel('k')
        plt.ylabel('Log KL Divergence')
        plt.yscale('log')  # Set y-axis to log scale
        # plt.legend()
        # Place legend outside the plot
        # plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.title('KL Divergence vs. k for Different n')
        plt.legend()
    else:
        plt.xlabel('n')
        plt.ylabel('Accuracy (%)')
        # plt.ylabel('Log Accuracy (%)')
        # plt.yscale('log')  # Set y-axis to log scale
        plt.title('Acccuracy Loss vs. Blocklength n for Different k')
        
        # plt.legend()
        # plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

        plt.subplot(2, 1, 2)  # Subplot for accuracy
        plt.xlabel('n')
        plt.ylabel('Log KL Divergence')
        # plt.yscale('log')  # Set y-axis to log scale
        plt.legend()
        plt.title('KL Divergence vs. Blocklength n for Different k')
        plt.legend()

    # Create a common legend outside the plot
    plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1.2), borderaxespad=0)

    plt.tight_layout()
    plt.show()
    if testing_options.test_with_joint_model:
        plt.savefig(f'{PLOT_PATH}/{testing_options.dataset}_nsamples_{testing_options.MAX_TEST_SAMPLES}_min_SNR_{min_SNR}_max_SNR_{max_SNR}_many_k_epsilon_n_test_{testing_options.epsilon_n}_SNRtrain_{testing_options.SNR}.png')
    else:
        plt.savefig(f'{PLOT_PATH}/{testing_options.dataset}_nsamples_{testing_options.MAX_TEST_SAMPLES}_min_SNR_{min_SNR}_max_SNR_{max_SNR}_many_k_epsilon_n_test_{testing_options.epsilon_n}_SNRtrain_{testing_options.SNR}.png')
        
    if PLOT_FIT:
        # Define a function to fit to the data (e.g., square root function)
        def sqrt_function(k, a, b):
            return a * np.sqrt(k) * np.sqrt(testing_options.epsilon_n) + b
        def n_function(n, a, b):
            return a * n + b
        # Plot the MSE values against k for each n
        for n, k_list in k_n_dict.items():
            popt, _ = curve_fit(sqrt_function, k_list, mse_values_dict[n])
            popt_n, _ = curve_fit(n_function, k_list, mse_values_dict[n])
            # Plot fitting curve
            # plt.subplot(1, 2, 2)  # Subplot for fit
            print(f'popt: {popt}')
            plt.plot(k_list, mse_values_dict[n], marker='o', linestyle='-', label=f'n = {n}')
            plt.plot(k_list, sqrt_function(np.array(k_list), *popt), linestyle='--', label=f'Sqrt Fit: n = {n}')
            plt.plot(k_list, n_function(np.array(k_list), *popt_n), linestyle='-', label=f'Linear Fit: n = {n}')

        plt.xlabel('k')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('MSE vs. k for Different n')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(f'{PLOT_PATH}/{testing_options.dataset}_nsamples_{testing_options.MAX_TEST_SAMPLES}_fit_epsilon_n_test_{testing_options.epsilon_n}_min_SNR_{min_SNR}_max_SNR_{max_SNR}_SNRtrain_{testing_options.SNR}.png')
        plt.close()
    