source venv/bin/activate
export CUDA_VISIBLE_DEVICES=1

echo "[INFO] training the Non-Covert Model..."

# train_feature_extractor=1
train_feature_extractor=0

# communication parameters
blocklength=512 # 512, 768, 1024, 2048, 4096

if [ $train_feature_extractor -eq 1 ]; then
    echo "[INFO] training the Feature Extractor..."
    n_epochs=60
    learning_rate=0.01 # for training the Feature Extractor
else
    echo "[INFO] training the Autoencoder..."
    n_epochs=60
    learning_rate=0.0003 # for training the Autoencoder
fi

batch_size=128
optimizer='Adam'
reconstruction_loss_fct='L1'
lambda_reconstruction=10
lambda_classification=1
lambda_power_constraint=0
lambda_covertness_constraint=0
lambda_l2_regularization=0
dataset='MNIST'
# dataset='CIFAR10'
# dataset='IID'
dataset_path='./datasets/'
epsilon_n=0.01 #0.01
# SNR in dB
SNR=1
channel_type='AWGN'

list_target_epsilon_n_dict[512]="10" 
list_target_epsilon_n_dict[768]="10" 
list_target_epsilon_n_dict[1024]="10" 
list_target_epsilon_n_dict[2048]="10" 
list_target_epsilon_n_dict[4096]="10"

target_kl_div=${list_target_epsilon_n_dict[$blocklength]} #0.05

# Declare the associative dictionary list_k where the key is the blocklength and the value is the list of k values
declare -A list_k_n_dict

# set the list of k values for the given blocklength, only O(n) k values
list_k_n_dict[512]="102 409 512"
list_k_n_dict[768]="153 614 768"
list_k_n_dict[1024]="204 819 1024"
list_k_n_dict[2048]="409 1638 2048" 
list_k_n_dict[4096]="819 3276 4096"

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
                        --dataset $dataset \
                        --dataset_path $dataset_path \
                        --blocklength $blocklength \
                        --A $A \
                        --C $C \
                        --k $k \
                        --SNR $SNR \
                        --epsilon_n $epsilon_n \
                        --target_kl_div $target_kl_div \
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