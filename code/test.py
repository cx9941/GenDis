import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json
from transformers import TrainingArguments
from init_parameters import ModelArguments, DataTrainingArguments, custom_args
import torch
from init_parameters import custom_args, ModelArguments, DataTrainingArguments
import os
os.environ["WANDB_PROJECT"] = "AAAI25"
from transformers import HfArgumentParser, TrainingArguments, set_seed
from utils.utils import create_and_prepare_model
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_json_file(json_file=custom_args.config_file)

# 设置随机种子
from transformers import set_seed
set_seed(training_args.seed)

# 模型与 Tokenizer 初始化
from utils.utils import create_and_prepare_model
model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)
tokenizer.padding_side = "left"

from typing import List
def generate_batch_responses(system_prompts: List[str], user_prompts: List[str],
                             max_new_tokens=128, temperature=0.7, top_p=0.9):
    assert len(system_prompts) == len(user_prompts)

    # 1. 构建 chat 格式
    chat_texts = []
    z = "A suitable category could be:"
    for s, u in zip(system_prompts, user_prompts):
        messages = [
            {"role": "system", "content": s},
            {"role": "user", "content": u},
            {"role": "assistant", "content": z},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False).replace(f'<|im_start|>assistant\n{z}<|im_end|>\n', f'<|im_start|>assistant\n{z}')
        chat_texts.append(prompt)

    # 2. 编码输入
    inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    # 3. 生成
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 4. 解码输出
    results = tokenizer.batch_decode(outputs)
    input_prompts = tokenizer.batch_decode(input_ids)
    return input_prompts, results

system_prompts = [
    "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided.",
    "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided.",
    "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided.",
    "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided.",
    "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided.",
    "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided.",
]
user_prompts = [
    "Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\nCould you help my figure out the exchange fee?",
    "Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\nI made a cash deposit to my account but i don't see it",
    "Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\nHello - I'm on the app and trying to purchase crypto. It's not going through. What am I doing wrong?",
    "Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\nIs there an extra charge to exchange different currencies?",
    "Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\nDo top-up limits exist?",
    "Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\nWhat is the procedure for depositing a virtual",
]

input_prompts, responses = generate_batch_responses(system_prompts, user_prompts)
for i, (input_prompt, response) in enumerate(zip(input_prompts, responses)):
    with open(f'../.trash/temp_outputs/input_{i}.txt', 'w') as w:
        w.write(input_prompt)
    with open(f'../.trash/temp_outputs/response_{i}.txt', 'w') as w:
        w.write(response)

results = {
    "inputs": input_prompts,
    "responses": responses,
}
import json
with open('../.trash/temp_outputs/result.json', 'w') as w:
    json.dump(results, w)