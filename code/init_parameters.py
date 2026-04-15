import json
from dataclasses import dataclass, field
from typing import Optional
import argparse
import os
os.environ["WANDB_DISABLED"]="true"

import time
# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="Meta-Llama-3.1-8B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )
    mode: str = field(
        default='train',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="demo",
        metadata={"help": "The preference dataset to use."},
    )
    data_root_dir: Optional[str] = field(
        default="../data",
        metadata={"help": "The preference dataset to use."},
    )
    rate: Optional[float] = field(
        default=0.25,
        metadata={"help": "The known proportion."},
    )
    labeled_ratio: Optional[float] = field(
        default=0.25,
        metadata={"help": "The known proportion."},
    )
    num_semi_warmup_epochs: Optional[float] = field(
        default=str,
        metadata={"help": "The known proportion."},
    )
    num_gen_warmup_epochs: Optional[float] = field(
        default=str,
        metadata={"help": "The known proportion."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(default="text", metadata={"help": "Dataset field to use as input text."})
    max_seq_length: Optional[int] = field(default=512)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, appends `eos_token_id` at the end of each sample being packed."},
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, tokenizers adds special tokens to each sample being packed."},
    )
    splits: Optional[str] = field(
        default="train,valid,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    is_semi: Optional[str] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    is_mlp: Optional[str] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    logit_adjustent: Optional[str] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    cca_loss_func: Optional[str] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    metric_dir: Optional[str] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    num_labels: Optional[int] = field(
        default=0,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    generate_dir: Optional[str] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    logs_dir: Optional[str] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    pretrain_output_dir: Optional[str] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    vector_dir: Optional[str] = field(
        default="our",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    shot_num: Optional[int] = field(
        default="-1",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    dataset_batch_size: Optional[int] = field(
        default=10000,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    cca_loss_weight: Optional[float] = field(
        default=1,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    cca_k: Optional[int] = field(
        default=10,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )

    imb_factor: Optional[int] = field(
        default=float,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    num_return_sequences: Optional[int] = field(
        default=float,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    num_iters_sk: Optional[int] = field(
        default=10,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )

    class_loss_weight: Optional[float] = field(
        default=1,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    com_loss_weight: Optional[float] = field(
        default=1,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    dis_loss_weight: Optional[float] = field(
        default=1,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )

    class_pseudo_loss_weight: Optional[float] = field(
        default=1,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    com_pseudo_loss_weight: Optional[float] = field(
        default=1,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    dis_pseudo_loss_weight: Optional[float] = field(
        default=1,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    cca_pseudo_loss_weight: Optional[float] = field(
        default=1,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )

    gen_loss_weight: Optional[float] = field(
        default=1,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    epsilon_sk: Optional[float] = field(
        default=1,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )
    linear_learning_rate: Optional[float] = field(
        default=5e-5,
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default="demo")
parser.add_argument("--data_root_dir", default="../data")
parser.add_argument("--rate", default=0.25, type=float)
parser.add_argument("--labeled_ratio", default=0.1, type=float)
parser.add_argument("--model_name_or_path", default="Qwen2.5-7B-Instruct")
parser.add_argument("--default_config", default="../configs/args.json")
parser.add_argument("--per_device_train_batch_size", default=32, type=int)
parser.add_argument("--per_device_eval_batch_size", default=32, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--gradient_accumulation_steps", default=2, type=int)
parser.add_argument("--eval_step_num", default=2, type=int)

parser.add_argument("--dataset_batch_size", default=100000, type=int)
parser.add_argument("--shot_num", default=-1, type=int)
parser.add_argument("--mode", default="train", choices=['train', 'eval-train', 'eval-test', 'eval-dev', 'gen-train', 'gen-dev', 'gen-test'], type=str)
parser.add_argument("--report_to", default="wandb", type=str)
parser.add_argument("--metric_for_best_model", default="kmeans_mlp_K-ACC", type=str)
parser.add_argument("--is_semi", default="semisurpervised", type=str, choices=['semisurpervised', 'nocurr-semisurpervised', 'nowas-semisurpervised'])
parser.add_argument("--is_mlp", default="mlp", type=str, choices=['gen', 'mlp'])
parser.add_argument("--cca_loss_func", default="log", type=str, choices=['mean', 'sum', 'log', 'sdl'])
parser.add_argument("--logit_adjustent", default="default", type=str, choices=['default', 'mlp'])

parser.add_argument("--gen_loss_weight", default=1.0, type=float)
parser.add_argument("--class_loss_weight", default=1.0, type=float)
parser.add_argument("--dis_loss_weight", default=1.0, type=float)

parser.add_argument("--com_loss_weight", default=1.0, type=float)
parser.add_argument("--cca_loss_weight", default=0.01, type=float)
parser.add_argument("--cca_k", default=16, type=int)
parser.add_argument("--class_pseudo_loss_weight", default=1.0, type=float)
parser.add_argument("--dis_pseudo_loss_weight", default=1.0, type=float)
parser.add_argument("--com_pseudo_loss_weight", default=1.0, type=float)
parser.add_argument("--cca_pseudo_loss_weight", default=0.0, type=float)

parser.add_argument("--num_semi_warmup_epochs", default=1, type=int)
parser.add_argument("--num_gen_warmup_epochs", default=2, type=int)
parser.add_argument("--num_train_epochs", default=4, type=int)

parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--linear_learning_rate", default=1e-4, type=float)


parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--epsilon_sk", default=0.1, type=float, help="epsilon for the Sinkhorn")
parser.add_argument("--imb_factor", default=1.0, type=float, help="imbalance factor of the data, default 1")
parser.add_argument("--num_return_sequences", default=4, type=int)

parser.add_argument("--gpu_id", default="0", type=str)
parser.add_argument("--run_name", default="NeurIPS-LLM4Open", type=str)

custom_args = parser.parse_args()
custom_args.cca_pseudo_loss_weight = 0.0
custom_args.num_iters_sk = 3
custom_args.epsilon_sk = 0.1
custom_args.imb_factor = 1.0
custom_args.num_return_sequences = int(4)

os.environ['CUDA_VISIBLE_DEVICES'] = custom_args.gpu_id

with open(custom_args.default_config, 'r') as r:
    config_content = json.load(r)

data_identify_id = f"{custom_args.dataset_name}_{custom_args.rate}_{custom_args.labeled_ratio}_{custom_args.shot_num}"


model_identify_id = f"{custom_args.model_name_or_path}/{custom_args.is_semi}_{custom_args.is_mlp}_{custom_args.cca_loss_func}_{custom_args.logit_adjustent}/{custom_args.gen_loss_weight}_{custom_args.class_loss_weight}_{custom_args.dis_loss_weight}/{custom_args.com_loss_weight}_{custom_args.cca_loss_weight}_{custom_args.cca_k}_{custom_args.class_pseudo_loss_weight}_{custom_args.dis_pseudo_loss_weight}_{custom_args.com_pseudo_loss_weight}_{custom_args.cca_pseudo_loss_weight}_{custom_args.num_return_sequences}/{custom_args.num_semi_warmup_epochs}_{custom_args.num_gen_warmup_epochs}_{custom_args.num_train_epochs}_{custom_args.learning_rate}_{custom_args.linear_learning_rate}"

pretrained_class_pseudo_loss_weight, pretrained_dis_pseudo_loss_weight, pretrained_com_pseudo_loss_weight, pretrained_cca_pseudo_loss_weight, pretrained_cca_loss_weight, pretrained_num_return_sequences = 0.0, 0.0, 0.0, 0.0, 0.0, 0
pretrained_num_train_epochs = pretrained_num_gen_warmup_epochs = custom_args.num_semi_warmup_epochs
pretrained_is_semi = 'semisurpervised'

pretrained_model_identify_id = f"{custom_args.model_name_or_path}/{pretrained_is_semi}_{custom_args.is_mlp}_{custom_args.cca_loss_func}_{custom_args.logit_adjustent}/{custom_args.gen_loss_weight}_{custom_args.class_loss_weight}_{custom_args.dis_loss_weight}/{custom_args.com_loss_weight}_{pretrained_cca_loss_weight}_{custom_args.cca_k}_{pretrained_class_pseudo_loss_weight}_{pretrained_dis_pseudo_loss_weight}_{pretrained_com_pseudo_loss_weight}_{pretrained_cca_pseudo_loss_weight}_{pretrained_num_return_sequences}/{custom_args.num_semi_warmup_epochs}_{pretrained_num_gen_warmup_epochs}_{pretrained_num_train_epochs}_{custom_args.learning_rate}_{custom_args.linear_learning_rate}"

custom_args.config_file = f"../configs/json/{custom_args.dataset_name}/{custom_args.dataset_name}_{custom_args.rate}_{custom_args.shot_num}/{data_identify_id}_{model_identify_id}/seed_{custom_args.seed}.json"

sample_num = json.load(open(f'{custom_args.data_root_dir}/data_statics.json', 'r'))

os.makedirs(f"../configs/json/{custom_args.dataset_name}/{custom_args.dataset_name}_{custom_args.rate}_{custom_args.shot_num}/{data_identify_id}_{model_identify_id}", exist_ok=True)

with open(custom_args.config_file, 'w') as w:
    config_content['data_root_dir'] = custom_args.data_root_dir
    config_content['dataset_name'] = custom_args.dataset_name
    config_content['learning_rate'] = custom_args.learning_rate
    config_content['linear_learning_rate'] = custom_args.linear_learning_rate
    config_content['is_semi'] = custom_args.is_semi
    config_content['cca_loss_func'] = custom_args.cca_loss_func
    config_content['is_mlp'] = custom_args.is_mlp
    config_content['logit_adjustent'] = custom_args.logit_adjustent
    config_content['rate'] = custom_args.rate
    config_content['labeled_ratio'] = custom_args.labeled_ratio
    config_content['num_labels'] = sample_num[custom_args.dataset_name]['num_labels']

    config_content['output_dir'] = f"../outputs/ckpts/{custom_args.dataset_name}/{data_identify_id}/{model_identify_id}/seed_{custom_args.seed}"
    config_content['pretrain_output_dir'] = f"../outputs/ckpts/{custom_args.dataset_name}/{data_identify_id}/{pretrained_model_identify_id}/seed_{custom_args.seed}"

    config_content['logs_dir'] = f"../outputs/logs/{custom_args.dataset_name}/{data_identify_id}/{model_identify_id}/seed_{custom_args.seed}"
    config_content['metric_dir'] = f"../outputs/metrics/{custom_args.dataset_name}/{data_identify_id}/{model_identify_id}/seed_{custom_args.seed}"
    config_content['generate_dir'] = f"../outputs/generate/{custom_args.dataset_name}/{data_identify_id}/{model_identify_id}/seed_{custom_args.seed}"
    config_content['vector_dir'] = f"../outputs/vectors/{custom_args.dataset_name}/{data_identify_id}/{model_identify_id}/seed_{custom_args.seed}"

    config_content['model_name_or_path'] = f"../pretrained_models/{custom_args.model_name_or_path}"    
    config_content['per_device_eval_batch_size'] = custom_args.per_device_eval_batch_size   
    config_content['per_device_train_batch_size'] = custom_args.per_device_train_batch_size    
    config_content['seed'] = custom_args.seed    
    config_content['mode'] = custom_args.mode
    config_content['metric_for_best_model'] = custom_args.metric_for_best_model
    config_content['dataset_batch_size'] = custom_args.dataset_batch_size
    config_content['gradient_accumulation_steps'] = custom_args.gradient_accumulation_steps
    config_content['num_train_epochs'] = custom_args.num_train_epochs
    config_content['num_semi_warmup_epochs'] = custom_args.num_semi_warmup_epochs
    config_content['num_gen_warmup_epochs'] = custom_args.num_gen_warmup_epochs
    
    config_content['dis_loss_weight'] = custom_args.dis_loss_weight
    config_content['com_loss_weight'] = custom_args.com_loss_weight
    config_content['class_loss_weight'] = custom_args.class_loss_weight
    config_content['gen_loss_weight'] = custom_args.gen_loss_weight
    config_content['cca_loss_weight'] = custom_args.cca_loss_weight

    config_content['dis_pseudo_loss_weight'] = custom_args.dis_pseudo_loss_weight
    config_content['com_pseudo_loss_weight'] = custom_args.com_pseudo_loss_weight
    config_content['class_pseudo_loss_weight'] = custom_args.class_pseudo_loss_weight
    config_content['cca_pseudo_loss_weight'] = custom_args.cca_pseudo_loss_weight


    config_content['cca_k'] = custom_args.cca_k

    config_content['num_iters_sk'] = custom_args.num_iters_sk
    config_content['epsilon_sk'] = custom_args.epsilon_sk
    config_content['imb_factor'] = custom_args.imb_factor
    config_content['num_return_sequences'] = custom_args.num_return_sequences

    # config_content['eval_strategy'] = 'steps'
    # config_content['save_strategy'] = 'steps'
    
    # config_content['eval_steps'] = max(10 * sample_num[custom_args.dataset_name][custom_args.rate]['train'] // (custom_args.per_device_train_batch_size * custom_args.gradient_accumulation_steps * len(custom_args.gpu_id.split(',')) * custom_args.eval_step_num), 5)
    
    # config_content['save_steps'] = max(10 * sample_num[custom_args.dataset_name][custom_args.rate]['train'] // (custom_args.per_device_train_batch_size * custom_args.gradient_accumulation_steps * len(custom_args.gpu_id.split(',')) * custom_args.eval_step_num), 5)

    config_content['eval_strategy'] = 'epoch'
    config_content['save_strategy'] = 'epoch'

    if custom_args.shot_num != -1:
        config_content['splits'] =  f"shot_{custom_args.shot_num}/train,shot_{custom_args.shot_num}/dev,test"

    json.dump(config_content, w)


os.makedirs(f"{config_content['metric_dir']}", exist_ok=True)
os.makedirs(f"{config_content['vector_dir']}", exist_ok=True)
os.makedirs(f"{config_content['logs_dir']}", exist_ok=True)
os.makedirs(f"{config_content['pretrain_output_dir']}", exist_ok=True)
if custom_args.num_gen_warmup_epochs < custom_args.num_train_epochs or 'gen' in custom_args.mode:
    os.makedirs(f"{config_content['generate_dir']}", exist_ok=True)

import logging
from datetime import datetime
import os  # 需要导入 os 模块

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # 添加了时分秒，并用下划线分隔
log_file = os.path.join(config_content['logs_dir'], f'{custom_args.mode}_{current_time}.log')

logging.basicConfig(
    filename=log_file, 
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)