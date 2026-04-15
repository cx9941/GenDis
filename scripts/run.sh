set -o errexit
gpu_id='0'

cd code

is_semi='semisurpervised'
labeled_ratio=0.1
learning_rate=5e-5
linear_learning_rate=1e-4

is_sk='sk'
num_iters_sk=3
epsilon_sk=0.1
imb_factor=1
num_return_sequences=3

model_name_or_path='Qwen2.5-7B-Instruct'

com_loss_weight=1 
gen_loss_weight=1 
class_loss_weight=1 
dis_loss_weight=1 
cca_loss_weight=0.01 

class_pseudo_loss_weight=1
dis_pseudo_loss_weight=1
com_pseudo_loss_weight=1
cca_pseudo_loss_weight=0.01

lambda_w=0.0
seed=1

num_semi_warmup_epochs=5
num_gen_warmup_epochs=10
num_train_epochs=12

per_device_train_batch_size=32
per_device_eval_batch_size=32


for seed in 2
do
for is_mlp in 'mlp'
do
for cca_loss_func in 'log'
do
for cca_k in 16
do
for dataset_name in 'banking' 'clinc' 'stackoverflow' 'mcid' 'hwu' 'ecdt'
do
for rate in 0.25 0.5 0.75
do
python main.py \
    --dataset_name $dataset_name \
    --rate $rate \
    --labeled_ratio $labeled_ratio \
    --gpu_id $gpu_id \
    --mode 'train' \
    --is_semi $is_semi \
    --is_mlp $is_mlp \
    --cca_k $cca_k \
    --cca_loss_weight $cca_loss_weight \
    --com_loss_weight $com_loss_weight \
    --gen_loss_weight $gen_loss_weight \
    --class_loss_weight $class_loss_weight \
    --dis_loss_weight $dis_loss_weight \
    --cca_loss_func $cca_loss_func \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --num_semi_warmup_epochs $num_semi_warmup_epochs \
    --num_gen_warmup_epochs $num_gen_warmup_epochs \
    --num_train_epochs $num_train_epochs \
    --model_name_or_path $model_name_or_path \
    --class_pseudo_loss_weight $class_pseudo_loss_weight \
    --dis_pseudo_loss_weight $dis_pseudo_loss_weight \
    --com_pseudo_loss_weight $com_pseudo_loss_weight \
    --cca_pseudo_loss_weight $cca_pseudo_loss_weight \
    --learning_rate $learning_rate \
    --linear_learning_rate $linear_learning_rate \
    --seed $seed \
    --num_iters_sk $num_iters_sk \
    --epsilon_sk $epsilon_sk \
    --imb_factor $imb_factor \
    --num_return_sequences $num_return_sequences


for mode in 'eval-train' 'eval-dev' 'eval-test'
do
python main.py \
    --dataset_name $dataset_name \
    --labeled_ratio $labeled_ratio \
    --rate $rate \
    --gpu_id $gpu_id \
    --mode $mode \
    --is_semi $is_semi \
    --is_mlp $is_mlp \
    --cca_k $cca_k \
    --cca_loss_weight $cca_loss_weight \
    --com_loss_weight $com_loss_weight \
    --gen_loss_weight $gen_loss_weight \
    --class_loss_weight $class_loss_weight \
    --dis_loss_weight $dis_loss_weight \
    --cca_loss_func $cca_loss_func \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --num_semi_warmup_epochs $num_semi_warmup_epochs \
    --num_gen_warmup_epochs $num_gen_warmup_epochs \
    --num_train_epochs $num_train_epochs \
    --model_name_or_path $model_name_or_path \
    --class_pseudo_loss_weight $class_pseudo_loss_weight \
    --dis_pseudo_loss_weight $dis_pseudo_loss_weight \
    --com_pseudo_loss_weight $com_pseudo_loss_weight \
    --cca_pseudo_loss_weight $cca_pseudo_loss_weight \
    --learning_rate $learning_rate \
    --linear_learning_rate $linear_learning_rate \
    --seed $seed \
    --num_iters_sk $num_iters_sk \
    --epsilon_sk $epsilon_sk \
    --imb_factor $imb_factor \
    --num_return_sequences $num_return_sequences

done
done
done
done
done
done
done