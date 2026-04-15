from scipy.spatial import distance as spatial_dist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from tqdm import tqdm
import dataclasses
import inspect
import warnings
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union
import datasets
import torch
import torch.nn as nn
import math
from accelerate.state import PartialState
from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from huggingface_hub.utils._deprecation import _deprecate_arguments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.configuration_utils import PretrainedConfig
from transformers import __version__
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.utils import check_torch_load_is_safe
CUSTOM_NAME = 'custom_module.bin'
import functools
from torch.nn.functional import softmax
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.trainer import _is_peft_model
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    SaveStrategy,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from deepspeed.runtime.zero.partition_parameters import GatheredParameters
import torch
import copy
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from torch.utils.data import DataLoader
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, EvalLoopOutput, has_length, denumpify_detensorize, speed_metrics
from transformers.trainer_pt_utils import EvalLoopContainer, IterableDatasetShard, find_batch_size, nested_detach
from transformers.utils import is_torch_xla_available
from trl.import_utils import is_liger_available, is_peft_available
from trl.trainer.sft_config import SFTConfig
from trl import SFTTrainer
from peft import PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
from transformers.utils import logging
import json
import os
from utils.utils import get_best_checkpoint
from transformers.integrations import (
    get_reporting_integration_callbacks,
    hp_params,
)
import torch.distributed as dist
from datasets import Dataset
import threading
from utils.utils import compute_uncertainty
import contextlib
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    EvalLoopContainer,
    IterableDatasetShard,
    LabelSmoother,
    LayerWiseDummyOptimizer,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
    remove_dummy_checkpoint,
)

from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_in_notebook,
    is_ipex_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    logging,
    strtobool,
)
from transformers.integrations.tpu import tpu_spmd_dataloader
from tqdm import tqdm
tqdm.pandas()

def _get_fsdp_ckpt_kwargs():
    # TODO: @AjayP13, @younesbelkada replace this check with version check at the next `accelerate` release
    if is_accelerate_available() and "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
    if IS_XLA_FSDPV2_POST_2_2:
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
else:
    IS_XLA_FSDPV2_POST_2_2 = False

from utils.utils import generate_prompt
from utils.cutom_llm import VLLMChatClient, ChatTranslateLLM

from utils.metric import hungarian_alignment_with_unlabeled, hungray_aligment

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)

import os
import re

from packaging import version
from transformers.utils import is_accelerate_available
from utils.metric import clustering_score
from models.sinkhorn_knopp import SinkhornKnopp, get_topk_mask

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        is_mlu_available,
        is_mps_available,
        is_npu_available,
        is_torch_version,
        is_xpu_available,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"

logger = logging.get_logger(__name__)

class OHTC_Trainer(SFTTrainer):

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SFTConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        semi_train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        gen_test_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = None,
        num_of_sequences: Optional[int] = None,
        chars_per_token: Optional[float] = None,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: Optional[int] = None,
        neftune_noise_alpha: Optional[float] = None,
        model_init_kwargs: Optional[Dict] = None,
        dataset_kwargs: Optional[Dict] = None,
        eval_packing: Optional[bool] = None,
        dataset_name: Optional[str] = None,
        rate: Optional[float] = None,
        labeled_ratio: Optional[float] = None,
        mode=None,
        model_name_or_path=None,
        data_root_dir=None,
        generate_dir=None,
        vector_dir=None,
        data_args=None
    ):
        self.model_name_or_path = model_name_or_path
        self.data_root_dir = data_root_dir
        self.dataset_name = dataset_name
        self.rate = rate
        self.mode = mode
        self.generate_dir = generate_dir
        self.vector_dir = vector_dir
        self.labeled_ratio = labeled_ratio
        self.eval_packing = eval_packing
        self.max_seq_length = max_seq_length
        self.dataset_text_field = dataset_text_field
        self.num_of_sequences = num_of_sequences
        self.chars_per_token = chars_per_token
        self.formatting_func = formatting_func
        self.dataset_kwargs = dataset_kwargs
        self.data_args = data_args

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,  # type: ignore
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            dataset_text_field=dataset_text_field,
            packing=packing,
            formatting_func=formatting_func,
            max_seq_length=max_seq_length,
            infinite=infinite,
            num_of_sequences=num_of_sequences,
            chars_per_token=chars_per_token,
            dataset_num_proc=dataset_num_proc,
            dataset_batch_size=dataset_batch_size,
            neftune_noise_alpha=neftune_noise_alpha,
            model_init_kwargs=model_init_kwargs,
            dataset_kwargs=dataset_kwargs,
            eval_packing=eval_packing,
        )


        if test_dataset is not None:
            _multiple = isinstance(test_dataset, dict)
            _test_datasets = test_dataset if _multiple else {"singleton": test_dataset}

            eval_packing = packing if eval_packing is None else eval_packing

            for _eval_dataset_name, _eval_dataset in _test_datasets.items():
                _test_datasets[_eval_dataset_name] = self._prepare_dataset(
                    _eval_dataset,
                    tokenizer,
                    eval_packing,
                    dataset_text_field,
                    max_seq_length,
                    formatting_func,
                    num_of_sequences,
                    chars_per_token,
                    **dataset_kwargs,
                )
            if not _multiple:
                test_dataset = _test_datasets["singleton"]

        if semi_train_dataset is not None:
            _multiple = isinstance(semi_train_dataset, dict)
            _test_datasets = semi_train_dataset if _multiple else {"singleton": semi_train_dataset}

            eval_packing = packing if eval_packing is None else eval_packing

            for _eval_dataset_name, _eval_dataset in _test_datasets.items():
                _test_datasets[_eval_dataset_name] = self._prepare_dataset(
                    _eval_dataset,
                    tokenizer,
                    eval_packing,
                    dataset_text_field,
                    max_seq_length,
                    formatting_func,
                    num_of_sequences,
                    chars_per_token,
                    **dataset_kwargs,
                )
            if not _multiple:
                semi_train_dataset = _test_datasets["singleton"]

        self.test_dataset = test_dataset
        self.semi_train_dataset = semi_train_dataset
        self.model.set_class(num_labels=self.data_args.num_labels, cca_k=self.data_args.cca_k)
        # self.chat_bot = ChatTranslateLLM()
        self.chat_bot = VLLMChatClient()


    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False

        if 't5' in tokenizer.name_or_path:
            id_token = tokenizer.encode('\nA suitable category could be:')[:-1]
            # norm_token = tokenizer.encode('\nNote: It was mapped to a normalized category:')[:-1]
        elif 'Qwen2' in tokenizer.name_or_path:
            id_token = tokenizer.encode('\nA suitable category could be:')[:-1]
            # norm_token = tokenizer.encode('\nNote: It was mapped to a normalized category:')[:-1]
        elif 'Qwen3' in tokenizer.name_or_path:
            id_token = tokenizer.encode('\nA suitable category could be:')[:-1]
            # norm_token = tokenizer.encode('\nNote: It was mapped to a normalized category:')[:-1]
        else:
            id_token = tokenizer.encode('\nA suitable category could be:')[2:-1]
            # norm_token = tokenizer.encode('\nNote: It was mapped to a normalized category:')[2:-1]


        def equal_list(x, y):
            if len(x) != len(y):
                return False
            for i,v in zip(x, y):
                if i != v:
                    return False
            return True

        def tokenize(element):

            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            token_type_ids = [[1 for j in i] for i in outputs["input_ids"]]

            for i in range(len(token_type_ids)):
                for j in range(len(token_type_ids[i]) + 1):
                    if j == len(token_type_ids[i]):
                        if equal_list(outputs['input_ids'][i][-len(id_token):], id_token):
                            token_type_ids[i][-1] = 2
                            for k in range(j + 1, len(token_type_ids[i])):
                                token_type_ids[i][k] = 3

                        # if equal_list(outputs['input_ids'][i][-len(norm_token):], norm_token):
                        #     token_type_ids[i][-1] = 5
                        #     for k in range(j + 1, len(token_type_ids[i])):
                        #         token_type_ids[i][k] = 6
                        #     for k in range(j-len(norm_token), j):
                        #         token_type_ids[i][k] = 4
                    else:
                        if equal_list(outputs['input_ids'][i][j-len(id_token):j], id_token):
                            token_type_ids[i][j] = 2
                            for k in range(j + 1, len(token_type_ids[i])):
                                token_type_ids[i][k] = 3

                        # if equal_list(outputs['input_ids'][i][j-len(norm_token):j], norm_token):
                        #     token_type_ids[i][j] = 5
                        #     for k in range(j + 1, len(token_type_ids[i])):
                        #         token_type_ids[i][k] = 6
                        #     for k in range(j-len(norm_token), j):
                        #         token_type_ids[i][k] = 4

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True
            
            if 'pseudo_labels' not in element:
                pseudo_labels = [-1 for i in range(len(element['label_id']))]
            else:
                pseudo_labels = element['pseudo_labels']

            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "token_type_ids": token_type_ids, "labels": element['label_id'], "pseudo_labels": pseudo_labels}

        map_kwargs = {
            "batched": True,
            "remove_columns": dataset.column_names,
            "batch_size": self.dataset_batch_size,
        }
        if isinstance(dataset, datasets.Dataset):
            map_kwargs["num_proc"] = self.dataset_num_proc  # this arg is not available for IterableDataset
        tokenized_dataset = dataset.map(tokenize, **map_kwargs)
        return tokenized_dataset
    
    def test(self, output_dir=None, metrics_dir=None):
        if 'train' in self.mode:
            eval_dataset = self.train_dataset
        elif 'dev' in self.mode:
            eval_dataset = self.eval_dataset
        elif 'test' in self.mode:
            eval_dataset = self.test_dataset
        else:
            assert False

        file_name = f"{self.mode}_{self.data_args.num_iters_sk}_{self.data_args.epsilon_sk}_{self.data_args.imb_factor}"

        # if not os.path.exists(f"{metrics_dir}/{self.mode}.json"):
        if output_dir is not None:
            checkpoint_path = get_best_checkpoint(output_dir)
            self._load_from_checkpoint(checkpoint_path)
        with torch.no_grad():
            metrics = self.evaluate(eval_dataset=eval_dataset, metric_key_prefix=self.mode)
        json.dump(metrics, open(f"{metrics_dir}/{file_name}.json", "w"))
        
        if metrics_dir is not None:
            metrics = json.load(open(f"{metrics_dir}/{file_name}.json", "r"))

            df = pd.DataFrame([metrics]).T.reset_index()[2:-3]
            df.columns = ['index', 'value']

            # df = self.handle_metrics_results(df)
            df['prefix'] = df['index'].apply(lambda x: x.split('_')[0])
            df['representation'] = df['index'].apply(lambda x: x.split('_')[1])
            df['train_method'] = df['index'].apply(lambda x: x.split('_')[2])
            df['metric'] = df['index'].apply(lambda x: x.split('_')[-1])
            df['method'] = df['representation'] + '_' + df['train_method'] 
            pivot_df = df.pivot(index=['method'], columns=['metric'], values=['value']).reset_index()
            pivot_df.columns = [i[0] for i in pivot_df.columns[:1]] + [i[1] for i in pivot_df.columns[1:]]
            pivot_df.to_csv(f"{metrics_dir}/{file_name}.csv", sep='\t', index=None)


    def test_generate(self, checkpoint_dir=None, generate_dir=None, test_dataset=None):
        if checkpoint_dir is not None:
            checkpoint_path = get_best_checkpoint(checkpoint_dir)
            self._load_from_checkpoint(checkpoint_path)

        tokenizer = self.tokenizer
        tokenizer.padding_side='left'
        model = self.model
        assistant_prompt = "A suitable category could be:"

        def generate_batch_responses(system_prompts, batch_user_prompts,
                                    max_new_tokens=128, temperature=0.7, top_p=0.9):
            # 构造 chat conversation 格式
            conversations = []
            for system_prompt, user_prompt in zip(system_prompts, batch_user_prompts):
                chat = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_prompt},
                ]
                # 转换为符合 ChatML 的 prompt 字符串
                promt = tokenizer.apply_chat_template(chat, tokenize=False)
                conversations.append(promt.replace('A suitable category could be:<|im_end|>\n', 'A suitable category could be:'))

            # 编码
            inputs = tokenizer(conversations, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        eos_token_id=tokenizer.eos_token_id,
                    )

            # 解码
            decoded = tokenizer.batch_decode(outputs)
            return decoded
    

        system_prompts = [
            "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided.",
            "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided.",
            "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided.",
            "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided.",
            "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided.",
            "You are an expert in open-domain text classification. Your task is to assign the most suitable and concise category label for any given input text, even if no predefined categories are provided.",
        ]
        batch_prompts = [
            "Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\nCould you help my figure out the exchange fee?",
            "Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\nI made a cash deposit to my account but i don't see it",
            "Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\nHello - I'm on the app and trying to purchase crypto. It's not going through. What am I doing wrong?",
            "Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\nIs there an extra charge to exchange different currencies?",
            "Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\nDo top-up limits exist?",
            "Please read the following text and determine the most appropriate category it belongs to.\n\nText to Classify:\nWhat is the procedure for depositing a virtual",
        ]
        results = generate_batch_responses(system_prompts, batch_prompts)
        for i in results:
            print(i + '\n')

    def generate(self, checkpoint_dir=None, generate_dir=None, test_dataset=None):
        logger.info("generate the outputs")
        print("generate the outputs")

        if checkpoint_dir is not None:
            checkpoint_path = get_best_checkpoint(checkpoint_dir)
            self._load_from_checkpoint(checkpoint_path)
        num_return_sequences=self.data_args.num_return_sequences
        generate_file = f"{generate_dir}/{self.mode}.csv"

        os.makedirs(generate_dir, exist_ok=True)

        tokenizer = self.tokenizer
        args = self.args

        if 'train' in self.mode:
            test_dataset = self.train_dataset if test_dataset is None else test_dataset
            origin_data = pd.read_csv(f"{self.data_args.data_root_dir}/{self.data_args.dataset_name}/train.tsv", sep='\t')
        elif 'dev' in self.mode:
            test_dataset = self.eval_dataset if test_dataset is None else test_dataset
            origin_data = pd.read_csv(f"{self.data_args.data_root_dir}/{self.data_args.dataset_name}/dev.tsv", sep='\t')
        elif 'test' in self.mode:
            test_dataset = self.test_dataset if test_dataset is None else test_dataset
            origin_data = pd.read_csv(f"{self.data_args.data_root_dir}/{self.data_args.dataset_name}/test.tsv", sep='\t')
        else:
            assert False

        if os.path.exists(generate_file):
            records = pd.read_csv(generate_file, sep='\t').drop_duplicates('text')
            origin_data = origin_data.drop_duplicates('text')
            if len(records) == len(origin_data):
                return
        

        _multiple = isinstance(test_dataset, dict)
        _test_datasets = test_dataset if _multiple else {"singleton": test_dataset}

        test_packing = False
        tokenizer.padding_side = "left" 
        for _test_dataset_name, _test_dataset in _test_datasets.items():
            _test_datasets[_test_dataset_name] = self._prepare_dataset(
                _test_dataset,
                tokenizer,
                test_packing,
                args.dataset_text_field,
                args.max_seq_length,
                None,
                args.num_of_sequences,
                args.chars_per_token,
                remove_unused_columns=args.remove_unused_columns if args is not None else True,
                **args.dataset_kwargs,
            )
        if not _multiple:
            test_dataset = _test_datasets["singleton"]
        test_dataset = test_dataset.add_column("idx", [i for i in range(len(test_dataset['input_ids']))])
        
        all_input_ids = []
        all_indices = []
        all_generated_ids = []

        model = self.model
        model.eval()
        test_dataloader = self.get_eval_dataloader(test_dataset)
        model, test_dataloader = self.accelerator.prepare(model, test_dataloader)
        unwrapped_model = self.accelerator.unwrap_model(model)
        self.accelerator.wait_for_everyone()
        with torch.cuda.amp.autocast():
            for index, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                with torch.inference_mode():
                    generated_tokens = unwrapped_model.generate(
                        input_ids=batch['input_ids'],
                        decoder_input_ids=batch['input_ids'] if 't5' in tokenizer.name_or_path else None,
                        attention_mask=batch['attention_mask'],
                        max_new_tokens=50,
                        do_sample=True,          # 禁用采样，启用贪婪搜索
                        # num_beams=1,              # 单束搜索，彻底贪婪
                        temperature=1.0,          # 温度无影响（温度>1是平滑采样，贪婪不需要）
                        top_k=50,                 # top_k和top_p在do_sample=False时不会生效，但可以写上以防万一
                        top_p=1.0,
                        num_return_sequences=num_return_sequences
                    )
                    generated_tokens = self.accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )
                    generated_tokens = self.accelerator.gather(generated_tokens).cpu()
                    gathered_input_ids = self.accelerator.gather(batch['input_ids']).cpu()

                self.accelerator.wait_for_everyone()
                all_generated_ids.extend(generated_tokens.tolist())
                all_input_ids.extend(gathered_input_ids.tolist())
                all_indices.extend(range(index * test_dataloader.total_batch_size,
                                 index * test_dataloader.total_batch_size + len(gathered_input_ids)))

        # ✅ 后处理统一进行
        if self.accelerator.is_main_process:
            outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in all_generated_ids]
            inputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in all_input_ids]

            final_outputs = [outputs[i:i + num_return_sequences] for i in range(0, len(outputs), num_return_sequences)]

            trimmed_outputs = [
                [item[item.find(inputs[idx]) + len(inputs[idx]):] if inputs[idx] in item else item
                for item in final_outputs[idx]]
                for idx in range(len(final_outputs))
            ]

            results = {
                "text": [],
                "label": [],
                "pred": [],
                # "pred2": []
            }

            for i, gen_list in enumerate(trimmed_outputs):
                origin_idx = all_indices[i]
                results['text'].append(origin_data['text'][origin_idx])
                results['label'].append(origin_data['label'][origin_idx])
                # results['pred'].append([_.split('\n')[0].strip(' \n') for _ in gen_list])
                results['pred'].append(gen_list)
                # results['pred2'].append([
                #     re.findall(r'It was mapped to a normalized category: (.*)', _)[0].strip(' \n')
                #     if re.findall(r'It was mapped to a normalized category: (.*)', _) else ''
                #     for _ in gen_list
                # ])

            # 写入文件
            records = pd.DataFrame(results)
            records = pd.merge(origin_data, records, on=['text', 'label'])
            records.to_csv(generate_file, index=None, sep='\t')

    def eval_without_metrics(self, dataset):
        logger.info("eval_without_metrics")
        print("eval_without_metrics")
        dataloader = self.get_eval_dataloader(dataset)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        self.model.eval()
        with torch.inference_mode():
            for step, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
                # Update the observed num examples
                batch_size = find_batch_size(inputs)
                # Prediction step
                losses, logits, labels = self.prediction_step(self.model, inputs, prediction_loss_only=False)

                if logits is not None:
                    logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                    if self.preprocess_logits_for_metrics is not None:
                        logits = self.preprocess_logits_for_metrics(logits, labels)
                    logits = self.gather_function((logits))
                    all_preds.add(logits)

                all_preds.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            all_preds = all_preds.get_arrays()
        return all_preds

    def get_semi_train_dataloader(self, apply_chat_template=True, tokenizer=None):
        current_epoch = int(self.state.epoch)
        if self.state.epoch < self.data_args.num_semi_warmup_epochs:
            train_dataloader = self.get_train_dataloader()
            if self.is_fsdp_xla_v2_enabled:
                train_dataloader = tpu_spmd_dataloader(train_dataloader)
            return train_dataloader

        ### 判别式伪标签
        ori_labels = pd.read_csv(f'../data/{self.dataset_name}/label/label.list', header=None)[0].tolist()
        
        known_labels = pd.read_csv(f'../data/{self.dataset_name}/label/label_{self.rate}.list', header=None)[0].tolist()
        ori_labels = known_labels + [i for i in ori_labels if i not in known_labels]
        known_lab = [i for i in range(len(known_labels))]

        df = pd.read_csv(f'../data/{self.dataset_name}/train.tsv', sep='\t')
        df['gold_ids'] = df['label'].apply(lambda x: ori_labels.index(x))

        # if not os.path.exists(f"{self.vector_dir}/{self.mode}_{current_epoch}_class_logits.pt"):
        _, mlp_class_logits, com_class_logits, class_hidden_states, mlp_hidden_states, com_hidden_states , token_type_ids, class_golds = self.eval_without_metrics(self.semi_train_dataset)
        all_hidden_states = {
                "class": class_hidden_states,
                "mlp": mlp_hidden_states,
                "com": com_hidden_states,
            }
        
        all_class_logits = {
                "mlp": mlp_class_logits,
                "com": com_class_logits,
            }
        
        torch.save(class_golds, f"{self.vector_dir}/{self.mode}_class_golds.pt")
        torch.save(all_hidden_states, f"{self.vector_dir}/{self.mode}_hidden_states.pt")
        torch.save(all_class_logits, f"{self.vector_dir}/{self.mode}_class_logits.pt")
        # else:
        #     class_golds = torch.load(f"{self.vector_dir}/{self.mode}_{current_epoch}_class_golds.pt")
        #     all_hidden_states = torch.load(f"{self.vector_dir}/{self.mode}_{current_epoch}_hidden_states.pt")
        #     all_class_logits = torch.load(f"{self.vector_dir}/{self.mode}_{current_epoch}_class_logits.pt")
        #     mlp_class_logits = all_class_logits['mlp']
        #     com_class_logits = all_class_logits['com']


        all_logits = {}
        propotypes = {}
        prev_labels = class_golds
        for key in all_hidden_states:
            propotypes[key] = np.zeros([len(ori_labels), all_hidden_states[key].shape[-1]])
            id_propotypes = np.stack([all_hidden_states[key][(class_golds!=-1) & (class_golds==i)].mean(axis=0) for i in known_lab])
            unlabeled_hidden_states = all_hidden_states[key][class_golds == -1]
            kmeans = KMeans(n_clusters=len(ori_labels), n_init=5)
            kmeans.fit(unlabeled_hidden_states)
            un_propotypes = kmeans.cluster_centers_
            distance = spatial_dist.cdist(id_propotypes, un_propotypes, 'euclidean')
            _, col_ind = linear_sum_assignment(distance)

            pro_l = []
            for i in range(len(col_ind)):
                pro_l.append(un_propotypes[col_ind[i]][:])
            
            pro_u = []
            for j in range(len(ori_labels)):
                if j not in col_ind:
                    pro_u.append(un_propotypes[j][:])
            proto_u = pro_l + pro_u   
            proto_u = torch.tensor(np.array(proto_u), dtype=torch.float)
            propotypes[key] = proto_u
            
            user_norm = all_hidden_states[key] / np.linalg.norm(all_hidden_states[key] + 1e-11, axis=1, keepdims=True)
            item_norm = propotypes[key] / np.linalg.norm(propotypes[key] + 1e-11, axis=1, keepdims=True)
            sim_matrix = np.dot(user_norm, item_norm.T)
            all_logits[key] = torch.tensor(sim_matrix)
            
        torch.save(propotypes, f"{self.vector_dir}/{self.mode}_propotypes.pt")

        if self.state.epoch > self.data_args.num_semi_warmup_epochs: ###除了第一次，后面都是优化这个分类层
            # all_logits['mlp'] = torch.tensor(mlp_class_logits)
            # all_logits['com'] = torch.tensor(com_class_logits)
            pseudo_mode_list = ['cluster', 'mlp']
            # self.model.score_dis(torch.tensor(mlp_hidden_states, device=self.model.device, dtype=dtype))
            # all_logits['com'] = self.model.score_com(torch.tensor(com_hidden_states, device=self.model.device, dtype=dtype))
        else:
            pseudo_mode_list = ['mlp', 'cluster']

        torch.save(all_logits, f"{self.vector_dir}/{self.mode}_logits.pt")
        
        if self.data_args.is_semi == 'nocurr-semisurpervised':
            topk = -1
        else:
            topk = int(min((self.state.epoch - self.data_args.num_semi_warmup_epochs + 1) / (self.data_args.num_gen_warmup_epochs - self.data_args.num_semi_warmup_epochs + 1), 1) * all_logits['mlp'].shape[0] / all_logits['mlp'].shape[1]) + 1

        for pseudo_mode in pseudo_mode_list:
            if pseudo_mode == 'mlp':
                pseudo_all_logits = {
                    "mlp": torch.tensor(mlp_class_logits),
                    "com": torch.tensor(com_class_logits)
                }
            else:
                pseudo_all_logits = all_logits
            
            pseudo_labels = self.adjust_logits(pseudo_all_logits, -1)
            print(f"The pseudo_mode is {pseudo_mode}")
            logger.info(f"The pseudo_mode is {pseudo_mode}")
            if len(pseudo_labels[pseudo_labels!=-1]) > 0:
                metric_result, ind = clustering_score(df['gold_ids'].values[pseudo_labels!=-1], pseudo_labels[pseudo_labels!=-1], known_lab)
                logger.info("The accuracy of pseudo_labels:")
                logger.info(metric_result)
                print(metric_result)

            pseudo_labels = self.adjust_logits(pseudo_all_logits, topk)
            pseudo_labels[class_golds!=-1] = -1
            df['pseudo_labels'] = pseudo_labels
            df['label_id'] = class_golds
            df.to_csv(f"{self.vector_dir}/{self.mode}_{pseudo_mode}_train_preds.csv", index=None, sep='\t')

            logger.info(f"The topk is {topk}, The num of pseudo_labels: is {len(pseudo_labels[pseudo_labels!=-1])}, The ratio is {len(pseudo_labels[pseudo_labels!=-1]) / len(pseudo_labels)}")
            print(f"The topk is {topk}, The num of pseudo_labels: is {len(pseudo_labels[pseudo_labels!=-1])}, The ratio is {len(pseudo_labels[pseudo_labels!=-1]) / len(pseudo_labels)}")
            
            if len(pseudo_labels[pseudo_labels!=-1]) > 0:
                metric_result, ind = clustering_score(df['gold_ids'].values[pseudo_labels!=-1], pseudo_labels[pseudo_labels!=-1], known_lab)
                logger.info(f"The accuracy of pseudo_labels of top {topk}:")
                logger.info(metric_result)
                print(metric_result)


        if self.state.epoch < self.data_args.num_gen_warmup_epochs:
            self.train_dataset = self.train_dataset.remove_columns("pseudo_labels")
            self.train_dataset = self.train_dataset.add_column("pseudo_labels", pseudo_labels)
            train_dataloader = self.get_train_dataloader()
            return train_dataloader
        
        ## 生成式伪标签
        from utils.utils import extract_labels

        origin_data = pd.read_csv(f'../data/{self.dataset_name}/train.tsv', sep='\t')
        labeled_data = pd.read_csv(f'../data/{self.dataset_name}/labeled_data/train_{self.labeled_ratio}.tsv', sep='\t')
        known_labels = pd.read_csv(f'../data/{self.dataset_name}/label/label_{self.rate}.list', header=None)[0].tolist()
        origin_data['label'] = origin_data.apply(lambda x: x['label'] if x['text'] in labeled_data['text'].tolist() and x['label'] in known_labels else None, axis=1)

        df = pd.read_csv(f"{self.vector_dir}/{self.mode}_{pseudo_mode}_train_preds.csv", sep='\t')
        df['text'] = origin_data['text']
        df['label'] = origin_data['label']
        df[f'content'] = df.progress_apply(lambda x: generate_prompt(x['text'], None), axis=1, result_type='expand')
        generate_dataset = Dataset.from_dict(df[['content', 'label_id', 'pseudo_labels']])
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        def map_process(samples):
            batch = []
            for conversation in samples["content"]:
                system_prompt = "You are an expert in text classification.\n"
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
                    # entry = tokenizer.apply_chat_template(conversation, tokenize=False)
                    entry = tokenizer.apply_chat_template(conversation, tokenize=False).replace('<|im_start|>assistant\nA suitable category could be:<|im_end|>\n', '<|im_start|>assistant\nA suitable category could be:')

                batch.append(entry)
            return {"content": batch}

        if apply_chat_template:
            generate_dataset = generate_dataset.map(
                    map_process,
                    batched=True,
                )
            
        args = self.args
        test_packing = False

        generate_dataset = self._prepare_dataset(
                generate_dataset,
                tokenizer,
                test_packing,
                args.dataset_text_field,
                args.max_seq_length,
                None,
                args.num_of_sequences,
                args.chars_per_token,
                remove_unused_columns=args.remove_unused_columns if args is not None else True,
                **args.dataset_kwargs,
            )
        
        current_epoch = int(self.state.epoch)
        
        if os.path.exists(f"{self.generate_dir}/train/{current_epoch}/{self.mode}_uncertainty.csv"):
            generate_df = pd.read_csv(f"{self.generate_dir}/train/{current_epoch}/{self.mode}_uncertainty.csv", sep='\t')
            generate_df['pred'] = generate_df['pred'].apply(eval)
            generate_df['ppl'] = generate_df['ppl'].apply(eval)
        else:

            self.generate(checkpoint_dir=None, generate_dir=f"{self.generate_dir}/train/{current_epoch}", test_dataset=generate_dataset)
            generate_df = pd.read_csv(f"{self.generate_dir}/train/{current_epoch}/{self.mode}.csv", sep='\t')
            generate_df['pred'] = generate_df['pred'].apply(eval)

            def extract_label(pred):
                for _ in pred.split('\n'):
                    _ = _.strip('\n ')
                    if len(_) > 0:
                        # if _.find('Note: It was mapped'):
                        #     _ = _[:_.find('Note: It was mapped')]
                        return _.strip('\n ')
                return ""

            generate_df['pred'] = generate_df['pred'].apply(lambda x: [extract_label(i) for i in x])

            all_ppls, all_wasserstein_dis = self.uncertain_quantify(generate_df)
            generate_df['ppl'] = all_ppls
            generate_df['wasserstein_dis'] = all_wasserstein_dis
            generate_df.to_csv(f"{self.generate_dir}/train/{current_epoch}/{self.mode}_uncertainty.csv", sep="\t", index=None)
        
        generate_df['selected_ppl'] = generate_df.apply(lambda row: row['ppl'][np.argmin(row['ppl'])], axis=1)
        generate_df['selected_ppl'] = (generate_df['selected_ppl'] - generate_df['selected_ppl'].min()) / (generate_df['selected_ppl'].max() - generate_df['selected_ppl'].min())
        generate_df['selected_pred'] = generate_df.apply(lambda row: row['pred'][np.argmin(row['ppl'])], axis=1)
        if self.data_args.is_semi != 'nowas-semisurpervised':
            generate_df['selected_pred'] = generate_df.apply(lambda x: x['selected_pred'] if x['wasserstein_dis'] < 0.01  else None, axis=1)
        generate_df['selected_pred'] = generate_df['selected_pred'].apply(lambda x: x if x is not None and len(x) > 0 else None)
        final_df = df[['text', 'label', 'pseudo_labels', 'label_id']]
        final_df['gold'] = generate_df['label']
        final_df['pred'] = [final_df['label'].tolist()[_] if final_df['label'].tolist()[_] is not None else pred for _, pred in enumerate(generate_df['selected_pred'].tolist())]
        final_df['pred'] = final_df.apply(lambda x: None if x['pseudo_labels'] == -1 and x['label_id'] == -1 else x['pred'], axis=1)
        final_df['pred'] = final_df['pred'].apply(lambda x: x.strip(' \n') if type(x) == type('') else x)
        
        final_df['final_label_id'] = final_df.apply(lambda x: x['pseudo_labels'] if x['label_id'] == -1 else x['label_id'], axis=1)
        final_group = final_df[~final_df['pred'].isna()][['text', 'final_label_id', 'pred', 'gold']].groupby('final_label_id').agg(list).reset_index()
        final_group = final_group[final_group['final_label_id'] != -1]
        print('normlize the prediction to L1 label')
        logger.info('normlize the prediction to L1 label')

        # final_group['norm_pred'] = final_group.progress_apply(lambda x: self.chat_bot.extract_new_label_with_retry(known_labels, x['text'], x['pred']) if x['final_label_id'] not in known_lab else known_labels[x['final_label_id']], axis=1)
        # final_group_df = final_group[['text', 'norm_pred']].explode('text')
        # final_df = pd.merge(final_df, final_group_df, on='text', how='left')
        # final_df[['text', 'gold', 'pred', 'norm_pred']].to_csv(f"{self.generate_dir}/train/{current_epoch}/{self.mode}_norm.csv", sep="\t", index=None)
        
        final_df[['text', 'gold', 'pred']].to_csv(f"{self.generate_dir}/train/{current_epoch}/{self.mode}_norm.csv", sep="\t", index=None)
        
        # final_df['content'] = final_df.progress_apply(lambda x: generate_prompt(x['text'], x['pred'], x['norm_pred']), axis=1, result_type='expand')
        # final_df = final_df[['text', 'gold', 'pred', 'label_id', 'pseudo_labels', 'content', 'norm_pred']]

        final_df['content'] = final_df.progress_apply(lambda x: generate_prompt(x['text'], x['pred']), axis=1, result_type='expand')
        final_df = final_df[['text', 'gold', 'pred', 'label_id', 'pseudo_labels', 'content']]


        filter_final_df = final_df[:]
        filter_final_df['pred'] = final_df.apply(lambda x: x['pred'] if (x['pseudo_labels'] not in known_lab and  x['pred'] not in known_labels) | (x['pseudo_labels'] in known_lab and  x['pred'] in known_labels) | (x['label_id'] != -1) else None, axis=1)

        # filter_final_df['content'] = filter_final_df.progress_apply(lambda x: generate_prompt(x['text'], x['pred'], x['norm_pred']), axis=1, result_type='expand')
        filter_final_df['content'] = filter_final_df.progress_apply(lambda x: generate_prompt(x['text'], x['pred']), axis=1, result_type='expand')


        final_train_dataset = Dataset.from_dict(filter_final_df[['content', 'label_id', 'pseudo_labels']])
        if apply_chat_template:
            final_train_dataset = final_train_dataset.map(
                    map_process,
                    batched=True,
                )
            
        self.train_dataset = self._prepare_dataset(
                final_train_dataset,
                tokenizer,
                test_packing,
                args.dataset_text_field,
                args.max_seq_length,
                None,
                args.num_of_sequences,
                args.chars_per_token,
                remove_unused_columns=args.remove_unused_columns if args is not None else True,
                **args.dataset_kwargs,
            )
        train_dataloader = self.get_train_dataloader()
        return train_dataloader

    def adjust_logits(self, all_logits, topk=20):
        # if self.state.epoch > self.data_args.num_semi_warmup_epochs:
        #     df = pd.read_csv(f"{self.vector_dir}/{self.mode}_train_preds.csv", sep='\t')
        #     prev_pseudo_labels = df['pseudo_labels'].values
        # else:
        #     prev_pseudo_labels = None

        # com_softmax_scores = torch.softmax(all_logits['mlp'], dim=-1)

        logits = all_logits['mlp']
        logits = logits.detach()

        pseudo_labels = torch.argmax(logits, dim=-1)

        if topk != -1:
            _, topk_indices = torch.topk(logits, k=topk, dim=0)  # 形状为[k, K]
            mask = torch.zeros_like(logits[:, 0], dtype=torch.bool)  # 形状[N]
            for class_idx in range(logits.size(1)):
                mask[topk_indices[:, class_idx]] = True
            pseudo_labels[~mask] = -1

        pseudo_labels = pseudo_labels.detach().cpu().numpy()

        return pseudo_labels


    def load_custom_params(self, custom_config_file):
        import torch
        import os

        if not os.path.exists(custom_config_file):
            print(f"[Warning] Custom parameter file not found: {custom_config_file}")
            return

        state = torch.load(custom_config_file, map_location="cpu")
        missing_modules = []
        skipped_keys = {}

        for k, v in state.items():
            if hasattr(self.model, k):
                module = getattr(self.model, k)
                model_state = module.state_dict()
                loadable_dict = {}
                skipped = []

                for param_k, param_v in v.items():
                    if param_k in model_state and model_state[param_k].shape == param_v.shape:
                        loadable_dict[param_k] = param_v
                    else:
                        skipped.append(param_k)

                if skipped:
                    skipped_keys[k] = skipped

                module.load_state_dict(loadable_dict, strict=False)
            else:
                missing_modules.append(k)

        if missing_modules:
            print(f"[Warning] The following modules were not found in self.model and were skipped: {missing_modules}")
        if skipped_keys:
            for mod, keys in skipped_keys.items():
                print(f"[Warning] In module '{mod}', the following keys were skipped due to shape mismatch: {keys}")
        else:
            print(f"[Info] Custom parameters successfully loaded from: {custom_config_file}")
        
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model

        self.load_custom_params(os.path.join(resume_from_checkpoint, CUSTOM_NAME))

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        # if multiple adapters exist, they get saved in sub directories
        adapter_subdirs = (
            [
                folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                and (
                    os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_WEIGHTS_NAME))
                    or os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_SAFE_WEIGHTS_NAME))
                )
            ]
            if os.path.isdir(resume_from_checkpoint)
            else []
        )

        if is_fsdp_ckpt and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_ckpt
            or adapter_subdirs
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if hasattr(self.args, "fp16") and self.args.fp16 is True:
                        logger.warning(
                            "Enabling FP16 and loading from smp < 1.10 checkpoint together is not supported."
                        )
                    check_torch_load_is_safe()
                    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)
                    # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
                    # release memory
                    del state_dict
            elif self.is_fsdp_enabled:
                load_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    resume_from_checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                else:
                    check_torch_load_is_safe()
                    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)

                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, False)
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        elif _is_peft_model(model):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            # TODO: in the future support only specific min PEFT versions
            if (hasattr(model, "active_adapter") or hasattr(model, "active_adapters")) and hasattr(
                model, "load_adapter"
            ):
                if os.path.exists(resume_from_checkpoint):
                    # For BC for older PEFT versions
                    if hasattr(model, "active_adapters"):
                        active_adapters = model.active_adapters
                        if len(active_adapters) > 1:
                            logger.warning("Multiple active adapters detected will only consider the first adapter")
                        active_adapter = active_adapters[0]
                    else:
                        active_adapter = model.active_adapter

                    if adapter_subdirs:
                        for subdir_name in adapter_subdirs:
                            peft_id = os.path.join(resume_from_checkpoint, subdir_name)
                            model.load_adapter(peft_id, subdir_name, is_trainable=(subdir_name == active_adapter))
                        model.set_adapter(active_adapter)
                    else:
                        model.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)

    def _load_best_model(self):
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        self.load_custom_params(os.path.join(self.state.best_model_checkpoint, CUSTOM_NAME))

        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(
                self.model_wrapped,
                self.state.best_model_checkpoint,
                load_module_strict=not _is_peft_model(self.model),
            )
        elif self.is_fsdp_enabled:
            load_result = load_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                model,
                self.state.best_model_checkpoint,
                **_get_fsdp_ckpt_kwargs(),
            )
        elif (
            os.path.exists(best_model_path)
            or os.path.exists(best_safe_model_path)
            or os.path.exists(best_adapter_model_path)
            or os.path.exists(best_safe_adapter_model_path)
        ):
            has_been_loaded = True
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(self.state.best_model_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=self.state.best_model_checkpoint,
                        tag=WEIGHTS_NAME,
                        partial=False,
                        load_optimizer=False,
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                        state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                    else:
                        check_torch_load_is_safe()
                        state_dict = torch.load(best_model_path, map_location="cpu", weights_only=True)

                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
            else:
                if _is_peft_model(model):
                    # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
                    # TODO: in the future support only specific min PEFT versions
                    if (hasattr(model, "active_adapter") or hasattr(model, "active_adapters")) and hasattr(
                        model, "load_adapter"
                    ):
                        # For BC for older PEFT versions
                        if hasattr(model, "active_adapters"):
                            active_adapter = model.active_adapters[0]
                            if len(model.active_adapters) > 1:
                                logger.warning("Detected multiple active adapters, will only consider the first one")
                        else:
                            active_adapter = model.active_adapter

                        if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                            try:
                                model.load_adapter(self.state.best_model_checkpoint, active_adapter)
                            except RuntimeError as exc:
                                if model.peft_config[active_adapter].is_prompt_learning:
                                    # for context: https://github.com/huggingface/peft/issues/2256
                                    msg = (
                                        "When using prompt learning PEFT methods such as "
                                        f"{model.peft_config[active_adapter].peft_type.value}, setting "
                                        "load_best_model_at_end=True can lead to errors, it is recommended "
                                        "to set this to False and to load the model manually from the checkpoint "
                                        "directory using PeftModel.from_pretrained(base_model, <path>) after training "
                                        "has finished."
                                    )
                                    raise RuntimeError(msg) from exc
                                else:
                                    raise
                            # Load_adapter has no return value present, modify it when appropriate.
                            from torch.nn.modules.module import _IncompatibleKeys

                            load_result = _IncompatibleKeys([], [])
                        else:
                            logger.warning(
                                "The intermediate checkpoints of PEFT may not be saved correctly, "
                                f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                                "Check some examples here: https://github.com/huggingface/peft/issues/96"
                            )
                            has_been_loaded = False
                    else:
                        logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
                        has_been_loaded = False
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    if self.args.save_safetensors and os.path.isfile(best_safe_model_path):
                        state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                    else:
                        check_torch_load_is_safe()
                        state_dict = torch.load(best_model_path, map_location="cpu", weights_only=True)

                    # If the model is on the GPU, it still works!
                    # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                    # which takes *args instead of **kwargs
                    load_result = model.load_state_dict(state_dict, False)
                if not is_sagemaker_mp_enabled() and has_been_loaded:
                    self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_INDEX_NAME)) or os.path.exists(
            os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)
        ):
            load_result = load_sharded_checkpoint(
                model, self.state.best_model_checkpoint, strict=is_sagemaker_mp_enabled()
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):

        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)


        # from deepspeed.runtime.zero.partition_parameters import GatheredParameters

        # unwrapped_model = self.accelerator.unwrap_model(self.model)
        # target_module = unwrapped_model.base_model.model.class_mlp  # 根据实际结构修改

        # saved_params = {}
        # with GatheredParameters([target_module.weight, target_module.bias], modifier_rank=None):
        #     print("class_mlp, weight:", target_module.weight.shape)
        #     print("class_mlp, bias:", target_module.bias.shape)
        #     saved_params['class_mlp.weight'] = target_module.weight.detach().cpu()
        #     saved_params['class_mlp.bias'] = target_module.bias.detach().cpu()

        # print(saved_params)




        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model


        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        
        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.


        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            epoch_dataloader = self.get_semi_train_dataloader(tokenizer=self.tokenizer)
            # epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    if not do_sync_step:
                        self.accelerator.gradient_state._set_sync_gradients(False)
                    else:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().cpu().item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = int(epoch + (step + 1 + steps_skipped) / steps_in_epoch)
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def get_extra_parameter_names(self, model) -> List[str]:
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        SPECIAL_MODULES = ["class_mlp", "score_dis", "score_com"]
        extra_parameters = [name for name, param in model.named_parameters() if any(i in name for i in SPECIAL_MODULES)]
        return extra_parameters

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            extra_parameters = self.get_extra_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in extra_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in extra_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.data_args.linear_learning_rate
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in extra_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in extra_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.data_args.linear_learning_rate
                }
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.

            if 'lr' in optimizer_grouped_parameters[0]:
                optimizer_kwargs.pop("lr")

            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)
        return self.optimizer

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

            ## 额外日志
            import re
            epoch = metrics.pop('epoch')
            key_columns = list(metrics.keys())
            for key in key_columns:
                if re.findall(r'|'.join(['loss', 'run_time', 'per_second', 'runtime']), key):
                    metrics.pop(key)
            df = pd.DataFrame([metrics]).T.reset_index()
            df.columns = ['index', 'value']
            metric_df = self.handle_metrics_results(df)
            logger.info(f'\nEpoch:{epoch}')
            logger.info(metric_df)
            logger.info('\n')
            ## 额外日志
    

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            if self.state.epoch <= self.data_args.num_semi_warmup_epochs:
                self._save_checkpoint(model, trial, new_output_dir=self.data_args.pretrain_output_dir)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def save_custom_params(self, path):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        custom_keys = ["class_mlp", "score_dis", "score_com", "cca_proj", "soft_cca"]
        custom_state = {}

        for k in custom_keys:
            if hasattr(unwrapped_model, k):
                module = getattr(unwrapped_model, k)
                # Gather all parameters inside the module
                with GatheredParameters(list(module.parameters()), modifier_rank=None):
                    # state_dict() must be called inside this context to get full weights
                    custom_state[k] = {
                        name: param.detach().cpu()
                        for name, param in module.state_dict().items()
                    }

        def is_main_process():
            # Return True if not using distributed, or distributed but rank 0
            if not torch.distributed.is_available():
                return True
            if not torch.distributed.is_initialized():
                return True
            return torch.distributed.get_rank() == 0

        if is_main_process():
            torch.save(custom_state, path)
            print(f"[Saved] Custom module parameters saved to: {path}")

    def _save_checkpoint(self, model, trial, new_output_dir=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        if new_output_dir is not None:
            run_dir = new_output_dir

        output_dir = os.path.join(run_dir, checkpoint_folder)
        print("Save models into", output_dir)
        # logger.info("Save models into", output_dir)
        self.save_model(output_dir, _internal_call=True)
        self.save_custom_params(os.path.join(output_dir, "custom_module.bin"))

        if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
            best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.should_save:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # we use mtime as default, filesystems without mtime support will be detected in `_sorted_checkpoints`
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


    # def _save_checkpoint(self, model, trial, new_output_dir=None):
    #     # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
    #     # want to save except FullyShardedDDP.
    #     # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

    #     # Save model checkpoint
    #     checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    #     if self.hp_search_backend is None and trial is None:
    #         self.store_flos()

    #     run_dir = self._get_output_dir(trial=trial)
    #     if new_output_dir is not None:
    #         run_dir = new_output_dir
    #     output_dir = os.path.join(run_dir, checkpoint_folder)
    #     self.save_model(output_dir, _internal_call=True)

    #     self.save_custom_params(os.path.join(output_dir, "custom_module.bin"))

    #     if not self.args.save_only_model:
    #         # Save optimizer and scheduler
    #         self._save_optimizer_and_scheduler(output_dir)
    #         # Save RNG state
    #         self._save_rng_state(output_dir)

    #     # Save the Trainer state
    #     if self.args.should_save:
    #         # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
    #         for cb in [
    #             cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
    #         ]:
    #             cb_name = cb.__class__.__name__
    #             cb_state = cb.state()
    #             if isinstance(self.state.stateful_callbacks[cb_name], list):
    #                 self.state.stateful_callbacks[cb_name].append(cb_state)
    #             else:
    #                 self.state.stateful_callbacks[cb_name] = cb_state
    #         self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

    #     if self.args.push_to_hub:
    #         self._push_from_checkpoint(output_dir)

    #     # Maybe delete some older checkpoints.
    #     if self.args.should_save:
    #         # Solely rely on numerical checkpoint id for rotation.
    #         # mtime is not reliable especially on some fuse fs in cloud environments.
    #         self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)




    def handle_metrics_results(self, df):
        df['prefix'] = df['index'].apply(lambda x: x.split('_')[0])
        # df['data_type'] = df['index'].apply(lambda x: x.split('_')[1])
        df['cluster'] = df['index'].apply(lambda x: x.split('_')[1])
        df['method'] = df['index'].apply(lambda x: x.split('_')[2])
        df['metric'] = df['index'].apply(lambda x: x.split('_')[3])

        metric_df = df.pivot(index=['prefix', 'cluster', 'method'], columns=['metric'], values=['value']).reset_index()
        metric_df.columns = [i[0] for i in metric_df.columns[:-6]] + [i[1] for i in metric_df.columns[-6:]]
        return metric_df
    

    def uncertain_quantify(self, generate_df, apply_chat_template=True):
        logger.info("uncertain_quantify the outputs")
        print("uncertain_quantify the outputs")
        df = generate_df.explode('pred')
        df['pred'] = df['pred'].apply(lambda x: x.strip(' '))
        df['content'] = df.progress_apply(lambda x: generate_prompt(x['text'], x['pred']), axis=1, result_type='expand')
        tokenizer = self.tokenizer
        args = self.args
        test_packing = False
        num_return_sequences = self.data_args.num_return_sequences

        def map_process(samples):
            batch = []
            for conversation in samples["content"]:
                system_prompt = "You are an expert in text classification.\n"
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
                    # entry = tokenizer.apply_chat_template(conversation, tokenize=False)
                    entry = tokenizer.apply_chat_template(conversation, tokenize=False).replace('<|im_start|>assistant\nA suitable category could be:<|im_end|>\n', '<|im_start|>assistant\nA suitable category could be:')

                batch.append(entry)
            return {"content": batch}

        df['label_id'] = -1
        uncertrain_dataset = Dataset.from_dict(df[['content', 'label_id']])
            
        if apply_chat_template:
            uncertrain_dataset = uncertrain_dataset.map(
                    map_process,
                    batched=True,
                )
            
        uncertrain_dataset = self._prepare_dataset(
                uncertrain_dataset,
                tokenizer,
                test_packing,
                args.dataset_text_field,
                args.max_seq_length,
                None,
                args.num_of_sequences,
                args.chars_per_token,
                remove_unused_columns=args.remove_unused_columns if args is not None else True,
                **args.dataset_kwargs,
            )
        uncertrain_dataloader = self.get_eval_dataloader(uncertrain_dataset)

        prompt_group = []
        tokens = []
        all_ppls = []
        all_wasserstein_dis = []
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.accelerator.wait_for_everyone()
        unwrapped_model.eval()

        with torch.cuda.amp.autocast():
            for index, batch in tqdm(enumerate(uncertrain_dataloader), total=len(uncertrain_dataloader)):
                with torch.inference_mode():
                    outputs = unwrapped_model(**batch)
                    logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
                    probs = softmax(logits, dim=-1)  # 变成概率
                    mask = batch['token_type_ids'] == 3  # shape: (batch_size, seq_len)
                    for i in range(batch['input_ids'].size(0)):  # 遍历 batch 内样本
                        positions = mask[i].nonzero(as_tuple=True)[0]  # 获取 token_type_ids == 3 的 token 索引
                        token_probs = probs[i, positions]  # shape: (num_label_tokens, vocab_size)
                        
                        prompt_group.append(token_probs.cpu())
                        tokens.append(batch['input_ids'][i, positions].cpu())

                        if len(prompt_group) == num_return_sequences:
                            ppl, wasserstein_dis = compute_uncertainty(prompt_group, tokens)
                            all_ppls.append(ppl.tolist())
                            all_wasserstein_dis.append(wasserstein_dis)
                            prompt_group = []
                            tokens = []

        return all_ppls, all_wasserstein_dis