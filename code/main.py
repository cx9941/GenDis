# from monitor import monitor_memory_limit
# monitor_memory_limit(400)

from init_parameters import custom_args, ModelArguments, DataTrainingArguments
import os
os.environ["WANDB_PROJECT"] = "AAAI25"
import glob

# os.environ["WANDB_DISABLED"]="true"
from transformers import HfArgumentParser, TrainingArguments, set_seed
from trainer import OHTC_Trainer
from utils.utils import create_and_prepare_model, create_datasets, get_latest_checkpoint, get_best_checkpoint
from utils.metric import Metrics
from utils.dataset_utils import SelfDataCollator
import pandas as pd
from trainer_callback import DatasetUpdateCallback

def main(model_args, data_args, training_args):

    file_name = f"{data_args.num_iters_sk}_{data_args.epsilon_sk}_{data_args.imb_factor}"

    print(f"{data_args.metric_dir}/{model_args.mode}_{file_name}.csv")

    if data_args.dataset_name != 'demo':
        if os.path.exists(f"{data_args.metric_dir}/{model_args.mode}_{file_name}.csv"):
            print(f"This model has been trained and tested on {model_args.mode}")
            exit()

        if model_args.mode == 'train' and glob.glob(f"{data_args.metric_dir}/eval-dev*"):
            print("This model has been trained")
            exit()

        # if model_args.mode == 'train' and get_best_checkpoint(training_args.output_dir):
        #     print("This model has been trained")
        #     exit()

    # Set seed for reproducibility
    set_seed(training_args.seed)
    print(data_args)
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = training_args.gradient_checkpointing and not model_args.use_unsloth
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": model_args.use_reentrant}
    # datasets
    datasets = create_datasets(tokenizer, data_args,training_args, apply_chat_template=True)
    # trainer
    print('Data Load Over')
    metric = Metrics(data_args, model_args, training_args)
    print('Metric Load Over')
    trainer = OHTC_Trainer(
        model=model,
        model_name_or_path=model_args.model_name_or_path,
        mode=model_args.mode,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=datasets['train-gen'] if 'gen' in model_args.mode else datasets['train'] if 'train' == model_args.mode else datasets['train-eval'],
        semi_train_dataset=datasets['train-semi'],
        eval_dataset=datasets['dev-gen'] if 'gen' in model_args.mode else datasets['dev'],
        # eval_dataset=datasets['dev-gen'] if 'gen' in model_args.mode else datasets['test'],
        test_dataset=datasets['test-gen'] if 'gen' in model_args.mode else datasets['test'],
        peft_config=peft_config,
        packing=data_args.packing,
        dataset_batch_size=data_args.dataset_batch_size,
        compute_metrics=metric.compute_metrics,
        preprocess_logits_for_metrics=metric.preprocess_logits_for_metrics,
        data_collator=SelfDataCollator(tokenizer=tokenizer, is_mlp=data_args.is_mlp, cca_loss_func=data_args.cca_loss_func, class_loss_weight=data_args.class_loss_weight, dis_loss_weight=data_args.dis_loss_weight, com_loss_weight=data_args.com_loss_weight, gen_loss_weight=data_args.gen_loss_weight, cca_loss_weight=data_args.cca_loss_weight, class_pseudo_loss_weight=data_args.class_pseudo_loss_weight, dis_pseudo_loss_weight=data_args.dis_pseudo_loss_weight, com_pseudo_loss_weight=data_args.com_pseudo_loss_weight, cca_pseudo_loss_weight=data_args.cca_pseudo_loss_weight).torch_call,
        dataset_kwargs={
            "append_concat_token": data_args.append_concat_token,
            "add_special_tokens": data_args.add_special_tokens,
        },
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
        dataset_name=data_args.dataset_name,
        rate=data_args.rate,
        labeled_ratio=data_args.labeled_ratio,
        generate_dir=data_args.generate_dir,
        vector_dir=data_args.vector_dir,
        data_root_dir=data_args.data_root_dir,
        data_args=data_args
    )

    # 使用时
    callback = DatasetUpdateCallback(trainer, None)
    trainer.add_callback(callback)

    print('Trainer Ready')
    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    if model_args.mode == 'train':
        # trainer.generate(generate_dir = data_args.generate_dir,test_dataset=datasets['train-gen'])

        resume_from_checkpoint = None
        resume_from_checkpoint = get_latest_checkpoint(training_args.output_dir)
        if resume_from_checkpoint is None:
            resume_from_checkpoint = get_latest_checkpoint(data_args.pretrain_output_dir)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model()
        # trainer.test(training_args.output_dir, data_args.metric_dir)
        print('Train Over')

    elif 'eval' in model_args.mode:
        # if data_args.train_ablation == 'nosft':
        #     trainer.test(None, data_args.metric_dir)
        # else:
        print("Load the checkpoint in ", training_args.output_dir)
        trainer.test(training_args.output_dir, data_args.metric_dir)

    elif 'gen' in model_args.mode:
        trainer.generate(training_args.output_dir, data_args.generate_dir)
        # trainer.generate(None, data_args.generate_dir)
        # trainer.test_generate(training_args.output_dir, data_args.generate_dir)
        # trainer.test_generate(None, data_args.generate_dir)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=custom_args.config_file)
    main(model_args, data_args, training_args)
