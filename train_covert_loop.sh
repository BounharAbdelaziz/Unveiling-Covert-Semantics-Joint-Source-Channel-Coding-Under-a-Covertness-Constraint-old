source venv/bin/activate
export CUDA_VISIBLE_DEVICES=2

# weither to train with O(sqrt(n)) or O(n) k values
# train_linear_covert=1
train_linear_covert=0

# train the Feature Extractor or the Autoencoder
# train_feature_extractor=1
train_feature_extractor=0

# communication parameters
blocklength=512 # 512, 768, 1024, 2048, 4096

# Declare the associative dictionary list_target_epsilon_n_dict where the key is the blocklength and the value is the targeted KL divergence
declare -A list_target_epsilon_n_dict

list_target_epsilon_n_dict[512]="0.020" 
list_target_epsilon_n_dict[768]="0.020"
list_target_epsilon_n_dict[1024]="0.010" 
list_target_epsilon_n_dict[2048]="0.001" 
list_target_epsilon_n_dict[4096]="0.005" 

if [ $train_feature_extractor -eq 1 ]; then
    echo "[INFO] training the Feature Extractor..."
    n_epochs=60
    learning_rate=0.01 # for training the Feature Extractor
    epsilon_n=1
else
    echo "[INFO] training the Autoencoder..."
    n_epochs=60
    # learning_rate=0.005 # for training the Autoencoder
    learning_rate=0.0003 # for training the Autoencoder
    lambda_power_constraint=10
    epsilon_n=0.01 #0.85 #0.1 #0.01
    target_kl_div=${list_target_epsilon_n_dict[$blocklength]} #0.05
    echo "[INFO] target_kl_div: $target_kl_div"
fi

batch_size=128
optimizer='Adam'
reconstruction_loss_fct='L1'
lambda_reconstruction=10
lambda_classification=1
lambda_covertness_constraint=0
lambda_l2_regularization=0
dataset='MNIST'
# dataset='CIFAR10'
# dataset='IID'
dataset_path='./datasets/'

# SNR in dB
SNR=1
channel_type='AWGN'

# Declare the associative dictionary list_k where the key is the blocklength and the value is the list of k values
declare -A list_k_n_dict

# set the list of k values for the given blocklength, based on the value of train_sqrt_n (i.e. whether to train with O(sqrt(n)) or O(n) k values)
if [ $train_linear_covert -eq 0 ]; then
    # O(sqrt(n)) k values, target_epsilon_n=0.2
    # list_k_n_dict[512]="1 3 4 6 7" 
    list_k_n_dict[512]="6" 
    list_k_n_dict[768]="1 2 9 12"
    list_k_n_dict[1024]="1 2 3 4 5 6 7 8 9 10"
    list_k_n_dict[2048]="2 4 5 8 10 11 12 14"
    list_k_n_dict[4096]="1 2 4 6 8 10 12 14 16 18 20"
else
    # O(n) k values
    list_k_n_dict[512]="102 409 512"
    list_k_n_dict[768]="153 614 768"
    list_k_n_dict[1024]="204 819 1024"
    list_k_n_dict[2048]="409 1638 2048" 
    list_k_n_dict[4096]="819 3276 4096"
fi

# the constant A influences the power constraint, the number of 1\'s in the codeword x^n
list_A=(1)
# the constant C influences the k parameter, i.e. k = C * sqrt(n*epsilon_n) #* sqrt(n)
list_C=(1)

# semantic encoder
input_dim_semantic_encoder=784
hidden_dim_semantic_encoder=256
n_hidden_semantic_encoder=2
do_quantize_U=1

# JSCC encoder
hidden_dim_jscc_encoder=$blocklength
n_hidden_jscc_encoder=2
output_dim_jscc_encoder=$blocklength
do_quantize=0

# JSCC decoder
input_dim_jscc_decoder=$blocklength
hidden_dim_jscc_decoder=$blocklength
n_hidden_jscc_decoder=3
do_quantize_U_hat=0

# classifier
hidden_dim_classifier=256
n_hidden_classifier=1
output_dim_classifier=10

# Iterate over parameter combinations
list_k=(${list_k_n_dict[$blocklength]})
if [ $train_linear_covert -eq 0 ]; then
    
    echo "[INFO] training with O(sqrt(n)) k values..."
    # we train covert model loss
    for k in "${list_k[@]}"; do
        for A in "${list_A[@]}"; do
            for C in "${list_C[@]}"; do

                echo "[INFO] running with blocklength: $blocklength, k: $k, A: $A, C: $C"

                    python3 main.py --train_covert_model --log_tensorboard \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --optimizer $optimizer \
                            --n_epochs $n_epochs \
                            --reconstruction_loss_fct $reconstruction_loss_fct \
                            --lambda_reconstruction $lambda_reconstruction \
                            --lambda_classification $lambda_classification \
                            --lambda_power_constraint $lambda_power_constraint \
                            --lambda_covertness_constraint $lambda_covertness_constraint \
                            --lambda_l2_regularization $lambda_l2_regularization \
                            --target_kl_div $target_kl_div \
                            --dataset $dataset \
                            --dataset_path $dataset_path \
                            --blocklength $blocklength \
                            --A $A \
                            --C $C \
                            --k $k \
                            --SNR $SNR \
                            --epsilon_n $epsilon_n \
                            --channel_type $channel_type \
                            --input_dim_semantic_encoder $input_dim_semantic_encoder \
                            --hidden_dim_semantic_encoder $hidden_dim_semantic_encoder \
                            --n_hidden_semantic_encoder $n_hidden_semantic_encoder \
                            --hidden_dim_jscc_encoder $hidden_dim_jscc_encoder \
                            --n_hidden_jscc_encoder $n_hidden_jscc_encoder \
                            --output_dim_jscc_encoder $output_dim_jscc_encoder \
                            --do_quantize $do_quantize \
                            --do_quantize_U $do_quantize_U \
                            --do_quantize_U_hat $do_quantize_U_hat \
                            --quantization_levels 0 1 \
                            --input_dim_jscc_decoder $input_dim_jscc_decoder \
                            --hidden_dim_jscc_decoder $hidden_dim_jscc_decoder \
                            --n_hidden_jscc_decoder $n_hidden_jscc_decoder \
                            --hidden_dim_classifier $hidden_dim_classifier \
                            --n_hidden_classifier $n_hidden_classifier \
                            --output_dim_classifier $output_dim_classifier &
            done
        done
    done
    # The wait command at the end ensures that the script waits for all background processes to finish before exiting.
    wait
else
    echo "[INFO] training with O(n) k values..."
    # we train non-covert model with power loss
    for k in "${list_k[@]}"; do
        for A in "${list_A[@]}"; do
            for C in "${list_C[@]}"; do

                echo "[INFO] running with blocklength: $blocklength, k: $k, A: $A, C: $C"

                    python3 main.py --log_tensorboard \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --optimizer $optimizer \
                            --n_epochs $n_epochs \
                            --reconstruction_loss_fct $reconstruction_loss_fct \
                            --lambda_reconstruction $lambda_reconstruction \
                            --lambda_classification $lambda_classification \
                            --lambda_power_constraint $lambda_power_constraint \
                            --lambda_covertness_constraint $lambda_covertness_constraint \
                            --lambda_l2_regularization $lambda_l2_regularization \
                            --target_kl_div $target_kl_div \
                            --dataset $dataset \
                            --dataset_path $dataset_path \
                            --blocklength $blocklength \
                            --A $A \
                            --C $C \
                            --k $k \
                            --SNR $SNR \
                            --epsilon_n $epsilon_n \
                            --channel_type $channel_type \
                            --input_dim_semantic_encoder $input_dim_semantic_encoder \
                            --hidden_dim_semantic_encoder $hidden_dim_semantic_encoder \
                            --n_hidden_semantic_encoder $n_hidden_semantic_encoder \
                            --hidden_dim_jscc_encoder $hidden_dim_jscc_encoder \
                            --n_hidden_jscc_encoder $n_hidden_jscc_encoder \
                            --output_dim_jscc_encoder $output_dim_jscc_encoder \
                            --do_quantize $do_quantize \
                            --do_quantize_U $do_quantize_U \
                            --do_quantize_U_hat $do_quantize_U_hat \
                            --quantization_levels 0 1 \
                            --input_dim_jscc_decoder $input_dim_jscc_decoder \
                            --hidden_dim_jscc_decoder $hidden_dim_jscc_decoder \
                            --n_hidden_jscc_decoder $n_hidden_jscc_decoder \
                            --hidden_dim_classifier $hidden_dim_classifier \
                            --n_hidden_classifier $n_hidden_classifier \
                            --output_dim_classifier $output_dim_classifier &
            done
        done
    done
    # The wait command at the end ensures that the script waits for all background processes to finish before exiting.
    wait
fi