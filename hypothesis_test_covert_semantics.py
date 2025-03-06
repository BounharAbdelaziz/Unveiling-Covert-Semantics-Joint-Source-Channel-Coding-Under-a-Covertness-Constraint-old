import numpy as np
import matplotlib.pyplot as plt

import torch

from model.trainer import Trainer
from model.trainer import TRAINING_MODE

import numpy as np
from options.base_options import Options

from sklearn.metrics import roc_curve, auc
from scipy.stats import entropy
import tikzplotlib

from tqdm import tqdm

def log_likelihood_ratio_vector(y_n, sigma, mean_under_H0, mean_under_H1):
    # More accurate calculation with correct handling of vector norms
    log_p_x_given_H0 = -0.5 * np.sum((y_n - mean_under_H0) ** 2 / sigma ** 2)
    log_p_x_given_H1 = -0.5 * np.sum((y_n - mean_under_H1) ** 2 / sigma ** 2)
    return log_p_x_given_H1 - log_p_x_given_H0

def run_test(k, blocklength, model_type, testing_options, plot_histogram=False):

    testing_options.k = k
    testing_options.blocklength = blocklength
    testing_options.output_dim_jscc_encoder = blocklength
    testing_options.hidden_dim_jscc_encoder = blocklength
    testing_options.input_dim_jscc_decoder = blocklength
    testing_options.hidden_dim_jscc_decoder = blocklength
    
    testing_options.train_covert_model = False
    testing_options.test_with_joint_model = False
    
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


    checkpoint_directory = f'./checkpoints/without-sk/{testing_options.dataset.upper()}/n/{testing_options.blocklength}/{model_type}/{is_qantized_sem}/{is_qantized_enc}/{is_qantized_dec}/C/{testing_options.C}/A/{testing_options.A}/k/{testing_options.k}/SNR/{str(testing_options.SNR)}_db/'
    # checkpoint_directory = f'./checkpoints/without-sk/{testing_options.dataset.upper()}/n/{testing_options.blocklength}/{model_type}/{is_qantized_sem}/{is_qantized_enc}/{is_qantized_dec}/C/{testing_options.C}/A/{testing_options.A}/k/{testing_options.k}/SNR/1.0_db/'
    
    testing_options.checkpoint_directory = checkpoint_directory
    print(f'[INFO] Checkpoint directory: {checkpoint_directory}')
    
    TRAIN_MODE = TRAINING_MODE.TEST
    # the maximum number of test samples
    MAX_TEST_SAMPLES = testing_options.MAX_TEST_SAMPLES
    # the flag to train the covert model
    TEST_COVERT_MODEL = testing_options.train_covert_model
    # Maximum power
    P_max = 1
    # the signal to noise ratio         
    snr = testing_options.SNR
    # Convert SNR from dB scale to linear scale
    snr_linear = 10 ** (snr / 10)
    # Calculate the noise standard deviation
    sigma = np.sqrt(P_max / (2 * snr_linear))
    # max power constraint
    if TEST_COVERT_MODEL:
        print(f'[INFO] Testing the covert model with k = O(sqrt(n))')
        maximum_power = testing_options.A * np.sqrt(testing_options.blocklength * testing_options.epsilon_n)
    else:
        maximum_power = testing_options.A * testing_options.blocklength
        print(f'[INFO] Testing the covert model with k = O(n)')

    print('[INFO] Initializing the trainer')
    # Create the trainer object
    trainer = Trainer(  testing_options,    # the training options
                        maximum_power,      # the maximum power constraint
                        TRAIN_MODE,         # Which model to train Train the model jointly or separately
                        LOAD_FORCED=False,
            )
    
    # set the model to evaluation mode
    trainer.semantic_covert_model.eval()
    # tracking variables
    z_0_vects = []
    z_1_vects = []
    # Evaluate the model on the test set
    with torch.no_grad():
        cpt = 0
        for images, labels in tqdm(trainer.testing_loader):

            images = images.to(trainer.device)
            labels = labels.to(trainer.device)
            
            _, encoded_information, _, _, noise = trainer.semantic_covert_model(images)
            noisy_codeword = encoded_information + noise

            # the output at the warden under the null hypothesis
            z_0 = noise.cpu().numpy()
            # the output at the warden under the alternative hypothesis
            z_1= noisy_codeword.cpu().numpy()
            
            # store the data
            z_0_vects.append(z_0)
            z_1_vects.append(z_1)
            
            if plot_histogram:
                # Compute histograms
                hist_0, bins_0 = np.histogram(z_0, bins=100)
                hist_1, bins_1 = np.histogram(z_1, bins=100)
                
                # Normalize the histograms
                hist_0 = hist_0 / np.sum(hist_0)
                hist_1 = hist_1 / np.sum(hist_1)

                plt.figure(figsize=(10, 5))
                plt.plot(bins_0[:-1], hist_0, label='Distribution under $\mathcal{H}=0$')
                plt.plot(bins_1[:-1], hist_1, label='Distribution under $\mathcal{H}=1$')
                plt.title('Output distribution under the two hypotheses')
                plt.xlabel('Bin')
                plt.ylabel('Frequency')
                plt.legend()
                plt.show()
                plt.savefig(f'./histogram_{model_type}.png')
                plt.close()
            
                # Filter out zero values
                non_zero_indices_0 = np.where(hist_0 != 0)[0]
                non_zero_indices_1 = np.where(hist_1 != 0)[0]

                # Find common non-zero indices
                common_indices = np.intersect1d(non_zero_indices_0, non_zero_indices_1)

                # Calculate the KL divergence based on non-zero values
                kl_div =  entropy(hist_1[common_indices], hist_0[common_indices])
            
                print(f'[INFO] KL divergence: {kl_div}')
           
            cpt += testing_options.batch_size
            if cpt == MAX_TEST_SAMPLES:
                break
            
    # Stack the data into numpy arrays
    z_0_vects, z_1_vects = np.vstack(z_0_vects), np.vstack(z_1_vects)

    # Combine data and labels
    y_all = np.vstack([z_0_vects, z_1_vects])
    labels = np.concatenate([np.zeros(len(z_0_vects)), np.ones(len(z_1_vects))])

    # Use the actual mean for hypothesis H1, we use the mean per feature (dimension-wise)
    mean_under_H1 = np.mean(z_1_vects, axis=0) 

    # Compute log-likelihood ratios for each vector
    log_LRT_all = np.array([log_likelihood_ratio_vector(y_n, sigma, mean_under_H0=0, mean_under_H1=mean_under_H1) for y_n in y_all])

    # Plot the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, log_LRT_all)
    roc_auc = auc(fpr, tpr)

    print(f'[INFO] Noise power (sigma): {sigma}')
    
    return fpr, tpr, roc_auc

if __name__ == "__main__":
    

    testing_options = Options().parse()
    
    plot_histogram = False
    
    # ----------------------------------------- Covert Model ----------------------------------------- #
    k = 6
    blocklength = 512
    model_type = 'covert'
    fpr_covert, tpr_covert, roc_auc_covert = run_test(k, blocklength, model_type, testing_options, plot_histogram)
    
    # ----------------------------------------- Linear Covert Model ----------------------------------------- #
    k = 409
    blocklength = 512
    model_type = 'non-covert-KL-training'
    fpr_linear_covert, tpr_linear_covert, roc_auc_linear_covert = run_test(k, blocklength, model_type, testing_options, plot_histogram)
    
    # ----------------------------------------- Non-Covert Model ----------------------------------------- #
    k = 102
    blocklength = 512
    model_type = 'non-covert'
    fpr_non_covert, tpr_non_covert, roc_auc_non_covert = run_test(k, blocklength, model_type, testing_options, plot_histogram)
    

    plt.figure()
    plt.plot(fpr_covert, tpr_covert, color='green', lw=2, label=f'Square-root covert model (area = {roc_auc_covert:.2f})')
    plt.plot(fpr_linear_covert, tpr_linear_covert, color='black', lw=2, label=f'Linear covert model (area = {roc_auc_linear_covert:.2f})')
    plt.plot(fpr_non_covert, tpr_non_covert, color='red', lw=2, label=f'Non-Covert model (area = {roc_auc_non_covert:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    plt.savefig(f'./roc_curve_n_{blocklength}.png')
    tikzplotlib.save(f'./roc_curve_n_{blocklength}.tex')
    
    plt.show()