from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import numpy as np
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch

def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

@dataclass
class SelfDataCollator():
    tokenizer: PreTrainedTokenizerBase
    is_mlp: Optional[str] = None
    cca_loss_func: Optional[str] = None
    pad_to_multiple_of: Optional[int] = None
    gen_loss_weight: Optional[float] = None
    class_loss_weight: Optional[float] = None
    dis_loss_weight: Optional[float] = None
    com_loss_weight: Optional[float] = None
    class_pseudo_loss_weight: Optional[float] = None
    dis_pseudo_loss_weight: Optional[float] = None
    com_pseudo_loss_weight: Optional[float] = None
    cca_loss_weight: Optional[float] = None
    cca_pseudo_loss_weight: Optional[float] = None

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }
        labels = batch.pop('labels')
        pseudo_labels = batch.pop('pseudo_labels')
        
        batch['labels'] = batch['input_ids'].clone()

        batch['class_golds'] = labels.to(labels.device)
        batch['pseudo_golds'] = pseudo_labels.to(pseudo_labels.device)

        batch['gen_loss_weight'] = torch.ones_like(labels).to(labels.device) * self.gen_loss_weight
        
        batch['class_loss_weight'] = torch.ones_like(labels).to(labels.device) * self.class_loss_weight
        batch['dis_loss_weight'] = torch.ones_like(labels).to(labels.device) * self.dis_loss_weight
        batch['com_loss_weight'] = torch.ones_like(labels).to(labels.device) * self.com_loss_weight
        batch['cca_loss_weight'] = torch.ones_like(labels).to(labels.device) * self.cca_loss_weight

        batch['class_pseudo_loss_weight'] = torch.ones_like(labels).to(labels.device) * self.class_pseudo_loss_weight
        batch['dis_pseudo_loss_weight'] = torch.ones_like(labels).to(labels.device) * self.dis_pseudo_loss_weight
        batch['com_pseudo_loss_weight'] = torch.ones_like(labels).to(labels.device) * self.com_pseudo_loss_weight
        batch['cca_pseudo_loss_weight'] = torch.ones_like(labels).to(labels.device) * self.cca_pseudo_loss_weight

        is_mlp_map = {'gen': 0, 'mlp': 1}
        cca_loss_func_map = {'mean': 0, 'sum': 1, 'log': 2, 'sdl': 3}
        
        batch['is_mlp'] = torch.ones_like(labels).to(labels.device) * is_mlp_map[self.is_mlp]
        batch['cca_loss_func'] = torch.ones_like(labels).to(labels.device) * cca_loss_func_map[self.cca_loss_func]

        return batch