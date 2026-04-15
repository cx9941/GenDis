import json
import os
from enum import Enum
from datasets import Dataset
import pandas as pd
import torch
from tqdm import tqdm
tqdm.pandas()
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import numpy as np
from peft import LoraConfig
import re
import random
from sklearn.metrics import classification_report
import torch.nn as nn

DEFAULT_CHATML_CHAT_TEMPLATE = "{% for message in messages %}\n{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% if loop.last and add_generation_prompt %}{{'<|im_start|>assistant\n' }}{% endif %}{% endfor %}"
DEFAULT_ZEPHYR_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


class ZephyrSpecialTokens(str, Enum):
    user = "<|user|>"
    assistant = "<|assistant|>"
    system = "<|system|>"
    eos_token = "</s>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

class ChatmlSpecialTokens(str, Enum):
    user = "<|im_start|>user"
    assistant = "<|im_start|>assistant"
    system = "<|im_start|>system"
    eos_token = "<|im_end|>"
    bos_token = "<s>"
    pad_token = "<pad>"

    @classmethod
    def list(cls):
        return [c.value for c in cls]

def format_nested_dict(d, indent=0):
    formatted_string = ""
    for key, value in d.items():
        formatted_string += "  " * indent + f"{key}:\n"
        if isinstance(value, dict):
            formatted_string += format_nested_dict(value, indent + 1)
        elif isinstance(value, list):
            for item in value:
                formatted_string += "  " * (indent + 1) + f"- {item}\n"
    return formatted_string

def generate_prompt(text, label):
    if label is None:
        content = f"Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\n{text}\nA suitable category could be:"
    else:
        content = f"Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\n{text}\nA suitable category could be: {label}"
        # if norm_label is not None:
        #     content += f"\nNote: It was mapped to a normalized category: {norm_label}"
    return content

def create_datasets(tokenizer, data_args, training_args, apply_chat_template=False):
    raw_datasets = DatasetDict()
    known_labels = pd.read_csv(f'../data/{data_args.dataset_name}/label/label_{data_args.rate}.list', sep='\t', header=None)[0].tolist()
    all_labels = pd.read_csv(f'../data/{data_args.dataset_name}/label/label.list', sep='\t', header=None)[0].tolist()
    split_set = ['train', 'dev', 'test', 'train-gen', 'dev-gen', 'test-gen', 'train-eval', 'train-semi']
    all_labels = known_labels + [i for i in all_labels if i not in known_labels]
    for split in split_set:
        
        df = pd.read_csv(f"../data/{data_args.dataset_name}/{split.split('-')[0]}.tsv", sep='\t')
        if 'train' == split or 'train-semi' == split or 'dev' == split:
            labeld_df = pd.read_csv(f"../data/{data_args.dataset_name}/labeled_data/{split.split('-')[0]}_{data_args.labeled_ratio}.tsv", sep='\t')
            df['label_id'] = df.apply(lambda x: all_labels.index(x['label']) if x['text'] in labeld_df['text'].tolist() and x['label'] in known_labels else -1, axis=1)
            df['label'] = df.apply(lambda x: None if x['label_id'] == -1 else x['label'], axis=1)

            if data_args.gen_loss_weight == 0 and data_args.class_loss_weight == 0 and 'train' == split:
                df = df[~df['label'].isna()]
                
        else:
            df['label_id'] = df['label'].apply(lambda x: all_labels.index(x))
        
        print("limit the max text length")
        df['text'] = df['text'].progress_apply(lambda x: tokenizer.decode(tokenizer(x)['input_ids'][:512], skip_special_tokens=True))

        if 'gen' in split:
            df[f'content'] = df.progress_apply(lambda x: generate_prompt(x['text'], None), axis=1, result_type='expand')
        else:
            df[f'content'] = df.progress_apply(lambda x: generate_prompt(x['text'], x['label']), axis=1, result_type='expand')

        dataset = Dataset.from_dict(df)
        raw_datasets[split] = dataset
    
    def map_process(samples):
        batch = []
        for conversation in samples["content"]:
            system_prompt = "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided."
            tag_seq = 'A suitable category could be:'
            question = conversation[:conversation.find(tag_seq)]
            answer = conversation[conversation.find(tag_seq):]

            if len(answer) > len(tag_seq):
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
                entry = tokenizer.apply_chat_template(conversation, tokenize=False)
            else:
                question = conversation[:conversation.find(tag_seq)]
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]  
                entry = tokenizer.apply_chat_template(conversation, tokenize=False).replace('<|im_start|>assistant\nA suitable category could be:<|im_end|>\n', '<|im_start|>assistant\nA suitable category could be:')
            batch.append(entry)
        return {"content": batch}
    
    if apply_chat_template:
        raw_datasets = raw_datasets.map(
                map_process,
                batched=True,
            )
    
    for split in split_set:
        raw_datasets[split] = raw_datasets[split] if split in raw_datasets else None

    print(f"Size of the train set: {len(raw_datasets['train'])}. Size of the validation set: {len( raw_datasets['dev'])}")
    print(f"A sample of train dataset: {raw_datasets['train'][0]}")
    return raw_datasets

def create_and_prepare_model(args, data_args, training_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )

        if 'Llama' in args.model_name_or_path:
            from models.modeling_llama import LlamaForCausalLM
            model = LlamaForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                torch_dtype=torch_dtype,
            )

        elif 'llava' in args.model_name_or_path:
            from models.modeling_llava import LlavaForConditionalGeneration
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                torch_dtype=torch_dtype,
            )

        elif 't5' in args.model_name_or_path:
            from models.modeling_t5 import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                torch_dtype=torch_dtype,
            )

        elif 'Qwen2' in args.model_name_or_path:
            from models.modeling_qwen2 import Qwen2ForCausalLM
            model = Qwen2ForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                torch_dtype=torch_dtype,
            )
        elif 'Qwen3' in args.model_name_or_path:
            from models.modeling_qwen3 import Qwen3ForCausalLM
            model = Qwen3ForCausalLM.from_pretrained(
                args.model_name_or_path,
                quantization_config=bnb_config,
                attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
                torch_dtype=torch_dtype,
            )


        # model.set_class(num_labels=data_args.num_labels, cca_k=data_args.cca_k)

    peft_config = None
    chat_template = None

    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    for name, param in model.named_parameters():
        if any(x in name for x in ["class_mlp", "score_dis", "score_com"]):
            print(name)
            param.requires_grad = True

    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
        # make embedding resizing configurable?
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )

    return model, peft_config, tokenizer


def load_model(args, data_args, training_args):
    if args.use_unsloth:
        from unsloth import FastLanguageModel
    bnb_config = None
    quant_storage_dtype = None

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and args.use_unsloth
    ):
        raise NotImplementedError("Unsloth is not supported in distributed training")

    if args.use_4bit_quantization:
        compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        quant_storage_dtype = getattr(torch, args.bnb_4bit_quant_storage_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.use_nested_quant,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

        if compute_dtype == torch.float16 and args.use_4bit_quantization:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
                print("=" * 80)
        elif args.use_8bit_quantization:
            bnb_config = BitsAndBytesConfig(load_in_8bit=args.use_8bit_quantization)

    if args.use_unsloth:
        # Load model
        model, _ = FastLanguageModel.from_pretrained(
            model_name=args.output_dir,
            max_seq_length=data_args.max_seq_length,
            dtype=None,
            load_in_4bit=args.use_4bit_quantization,
        )
    else:
        torch_dtype = (
            quant_storage_dtype if quant_storage_dtype and quant_storage_dtype.is_floating_point else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.output_dir,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
            torch_dtype=torch_dtype,
        )

    peft_config = None
    chat_template = None
    if args.use_peft_lora and not args.use_unsloth:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules,
        )

    special_tokens = None
    chat_template = None
    if args.chat_template_format == "chatml":
        special_tokens = ChatmlSpecialTokens
        chat_template = DEFAULT_CHATML_CHAT_TEMPLATE
    elif args.chat_template_format == "zephyr":
        special_tokens = ZephyrSpecialTokens
        chat_template = DEFAULT_ZEPHYR_CHAT_TEMPLATE

    if special_tokens is not None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir,
            pad_token=special_tokens.pad_token.value,
            bos_token=special_tokens.bos_token.value,
            eos_token=special_tokens.eos_token.value,
            additional_special_tokens=special_tokens.list(),
            trust_remote_code=True,
        )
        tokenizer.chat_template = chat_template
        # make embedding resizing configurable?
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

    if args.lora_target_modules == "all-linear":
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "lm_head"
            # 可以根据你模型结构添加
        ]
    else:
        target_modules = args.lora_target_modules.split(",")

    if args.use_unsloth:
        # Do model patching and add fast LoRA weights
        model = FastLanguageModel.get_peft_model(
            model,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            target_modules=target_modules,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            random_state=training_args.seed,
            max_seq_length=data_args.max_seq_length,
        )

    return model, peft_config, tokenizer


def get_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if re.match(r'checkpoint-\d+', d)]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)
    latest_checkpoint = os.path.join(output_dir, checkpoints[0])
    return latest_checkpoint

def get_best_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if re.match(r'checkpoint-\d+', d)]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]), reverse=False)
    latest_checkpoint = os.path.join(output_dir, checkpoints[0])
    return latest_checkpoint


import numpy as np
from sklearn.cluster import KMeans

def kemans(X):
    kmeans = KMeans(n_clusters=4)  # 设置聚类数为4
    kmeans.fit(X)  # 拟合数据
    y_kmeans = kmeans.predict(X)  # 预测聚类标签
    return y_kmeans


def extract_labels(text_list):
    extracted_labels = []
    for text in text_list:
        text = re.sub(r'_+', ' ', text)
        text = re.sub(r'\b(A suitable category could be:|Text to Classify:)\b', '', text, flags=re.IGNORECASE)
        candidates = re.findall(r'\b[a-zA-Z0-9_]+\b', text)
        labels = [c for c in candidates if len(c) > 2]
        extracted_labels.extend(labels)
    seen = set()
    unique_labels = []
    for label in extracted_labels:
        if label not in seen:
            seen.add(label)
            unique_labels.append(label)
    return '_'.join(unique_labels)




import torch
import ot
import torch.nn.functional as F

def compute_wasserstein_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    p, q: [seq_len, vocab_size]，分别表示两个序列中每个 token 的分布
    """
    cost_matrix = torch.cdist(p, q, p=2)  # [seq_len, seq_len]
    a = torch.ones(p.size(0)) / p.size(0)
    b = torch.ones(q.size(0)) / q.size(0)
    distance = ot.emd2(a.numpy(), b.numpy(), cost_matrix.detach().cpu().numpy())
    return distance

def compute_uncertainty(prompt_group, tokens):
    results = {}
    num_seqs = len(prompt_group)
    wasserstein_sum = 0.0

    # 1. Wasserstein 距离
    for i in range(num_seqs):
        for j in range(i + 1, num_seqs):
            wasserstein_sum += compute_wasserstein_distance(prompt_group[i], prompt_group[j])
    avg_wasserstein = wasserstein_sum / (num_seqs * (num_seqs - 1) // 2)

    # 2. PPL：每个 sample 单独计算 PPL
    ppls = []
    for idx_sample, sample in enumerate(prompt_group):
        neg_log_probs = []
        for idx_p in range(len(sample)):
            p = sample[idx_p]
            token_id = tokens[idx_sample][idx_p]
            prob = p[token_id].clamp(min=1e-8)  # 防止 log(0)
            neg_log_probs.append(-torch.log(prob))
        avg_neg_log_prob = torch.stack(neg_log_probs).mean()
        ppls.append(torch.exp(avg_neg_log_prob).item())
    avg_neg_log_prob = torch.tensor(ppls).log()

    return avg_neg_log_prob, avg_wasserstein