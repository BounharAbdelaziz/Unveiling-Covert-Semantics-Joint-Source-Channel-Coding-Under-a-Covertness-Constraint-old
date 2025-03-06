source venv/bin/activate
export CUDA_VISIBLE_DEVICES=2

MAX_TEST_SAMPLES=10000
batch_size=10000
reconstruction_loss_fct='L1'
lambda_reconstruction=10
lambda_classification=1
lambda_power_constraint=10
lambda_covertness_constraint=0
lambda_l2_regularization=0
# dataset='IID'
dataset='MNIST'
# dataset='CIFAR10'
dataset_path='./datasets/'
epsilon_n=0.002
        
# communication parameters
blocklength=2048 #512
# the constant A influences the power constraint, the number of 1\'s in the codeword x^n
A=1
# the constant C influences the k parameter, i.e. k = C * sqrt(n)
C=1
SNR=1
channel_type='AWGN'

list_target_epsilon_n_dict[512]="0.020" 
list_target_epsilon_n_dict[768]="0.020"
list_target_epsilon_n_dict[1024]="0.010" 
list_target_epsilon_n_dict[2048]="0.001" 
list_target_epsilon_n_dict[4096]="0.005" 

target_kl_div=${list_target_epsilon_n_dict[$blocklength]}
echo "[INFO] target_kl_div: $target_kl_div"

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

# python3 compare_coverts_acc_k_n.py \

python3 hypothesis_test_covert_semantics.py \
                --batch_size $batch_size \
                --MAX_TEST_SAMPLES $MAX_TEST_SAMPLES \
                --lambda_reconstruction $lambda_reconstruction \
                --lambda_classification $lambda_classification \
                --lambda_power_constraint $lambda_power_constraint \
                --lambda_covertness_constraint $lambda_covertness_constraint \
                --lambda_l2_regularization $lambda_l2_regularization \
                --dataset $dataset \
                --dataset_path $dataset_path \
                --blocklength $blocklength \
                --target_kl_div $target_kl_div \
                --A $A \
                --C $C \
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
                --output_dim_classifier $output_dim_classifier