from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging
from dataclasses import dataclass
from transformers import LlamaPreTrainedModel, LlamaModel, AutoTokenizer
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from .cca_model import CCAProjection
from .sdl_loss import SoftCCALoss
from .utils import triplet_ranking_loss

from transformers.models.llama.modeling_llama import LlamaModel, KwargsForCausalLM, can_return_tuple, auto_docstring
from transformers.processing_utils import Unpack

from transformers.utils import ModelOutput
@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mlp_class_logits: torch.FloatTensor = None
    com_class_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    class_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    mlp_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    com_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    token_type_ids: Optional[Tuple[torch.FloatTensor, ...]] = None
    class_golds: Optional[Tuple[torch.FloatTensor, ...]] = None



logger = logging.get_logger(__name__)

# tok = AutoTokenizer.from_pretrained('../pretrained_models/Meta-Llama-3.1-8B-Instruct')


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.propotype = {}

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    def set_class(self, num_labels, cca_k=10):
        self.class_mlp = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.hyper_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.score_dis = nn.Linear(self.hidden_size, num_labels, bias=True)
        self.score_com = nn.Linear(2 * cca_k, num_labels, bias=True)
        self.cca_k = cca_k

        self.cca_proj = CCAProjection(self.hidden_size, self.hidden_size, cca_k)
        self.soft_cca = SoftCCALoss(self.cca_k)

        self.soft_cca.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.hyper_proj.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.class_mlp.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.score_dis.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.score_com.to(device=self.device, dtype=next(self.parameters()).dtype)
        self.cca_proj.to(device=self.device, dtype=next(self.parameters()).dtype)

        # 手动 register_modules
        self.add_module("class_mlp", self.class_mlp)
        self.add_module("hyper_proj", self.hyper_proj)
        self.add_module("score_dis", self.score_dis)
        self.add_module("score_com", self.score_com)
        self.add_module("cca_proj", self.cca_proj)
        self.add_module("soft_cca", self.soft_cca)


    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if 'token_type_ids' not in kwargs:
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values
            )
        final_outputs = self.compute_loss(input_ids, logits, hidden_states, kwargs)

        loss = final_outputs['loss']

        return CausalLMOutputWithPast(
            loss=loss,
            logits=final_outputs['logits'],
            mlp_class_logits=final_outputs['mlp_class_logits'],
            com_class_logits=final_outputs['com_class_logits'],
            past_key_values=outputs.past_key_values,
            class_hidden_states=final_outputs['class_hidden_states'],
            mlp_hidden_states=final_outputs['mlp_hidden_states'],
            com_hidden_states=final_outputs['com_hidden_states'],
            token_type_ids=kwargs['token_type_ids'],
            class_golds=kwargs['class_golds'],
        )

    
    def compute_loss(self, input_ids, logits, hidden_states, kwargs):
        loss = None
        mlp_class_logits = None
        com_class_logits = None
        loss_kwargs = {'dtype': hidden_states.dtype, 'device': hidden_states.device, 'requires_grad': True}
        loss_gen = torch.tensor(0.0, **loss_kwargs)
        loss_dis = torch.tensor(0.0, **loss_kwargs)
        loss_com = torch.tensor(0.0, **loss_kwargs)
        loss_class = torch.tensor(0.0, **loss_kwargs)
        loss_dis_pseudo = torch.tensor(0.0, **loss_kwargs)
        loss_com_pseudo = torch.tensor(0.0, **loss_kwargs)
        loss_class_pseudo = torch.tensor(0.0, **loss_kwargs)
        loss_cca = torch.tensor(0.0, **loss_kwargs)
        loss_triplet_ranking_loss = torch.tensor(0.0, **loss_kwargs)

        loss_fct = CrossEntropyLoss(ignore_index=-100)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)

        if kwargs['gen_loss_weight'][0].item() != 0:
            gen_labels = input_ids.clone()
            gen_labels[kwargs['token_type_ids']<=0] = -100
            shift_gen_labels = gen_labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
            loss_gen = loss_fct(shift_logits, shift_gen_labels)


        if kwargs['class_loss_weight'][0].item() != 0:
            if kwargs['class_golds'][kwargs['class_golds']!=-1].shape[0] > 0:
                shift_class_logits = logits[kwargs['class_golds'] !=-1, :-1, :].contiguous()
                shift_class_logits = shift_class_logits.view(-1, self.config.vocab_size)

                gen_labels = input_ids.clone()
                gen_labels[kwargs['token_type_ids']<=2] = -100

                gen_labels = gen_labels[kwargs['class_golds'] !=-1]

                shift_gen_labels = gen_labels[..., 1:].contiguous().view(-1).to(shift_class_logits.device)
                loss_class = loss_fct(shift_class_logits, shift_gen_labels)

        if kwargs['class_pseudo_loss_weight'][0].item() != 0:
            shift_class_pseudo_logits = logits[kwargs['class_golds'] ==-1, :-1, :].contiguous()
            shift_class_pseudo_logits = shift_class_pseudo_logits.view(-1, self.config.vocab_size)
            gen_labels = input_ids.clone()
            gen_labels[kwargs['token_type_ids']<=2] = -100
            gen_labels = gen_labels[kwargs['class_golds'] ==-1]
            shift_gen_labels = gen_labels[..., 1:].contiguous().view(-1).to(shift_class_pseudo_logits.device)
            if len(shift_gen_labels[shift_gen_labels!=-100]) > 0:
                loss_class_pseudo = loss_fct(shift_class_pseudo_logits, shift_gen_labels)

        # shift_hidden_states = hidden_states[:,:-1,:].contiguous().to(shift_logits.device)
        # shift_token_type_ids = kwargs['token_type_ids'][...,1:].contiguous().to(shift_logits.device)
        # class_hidden_states = shift_hidden_states[shift_token_type_ids==1]

        class_hidden_states, mlp_hidden_states, com_hidden_states = None, None, None
        
        if (kwargs['token_type_ids']==2).sum() != 0:
            class_hidden_states = hidden_states[kwargs['token_type_ids']==2]
    

        # self.propotype_update(class_hidden_states, kwargs['class_golds'], num_samples=kwargs['vos_neg_sample'].mean().int().item())
        
        
        if kwargs['is_mlp'][0].item() == 0: ## gen
            class_logits = self.score_dis(class_hidden_states)
            if kwargs['dis_loss_weight'][0].item() != 0:
                if kwargs['class_golds'][kwargs['class_golds']!=-1].shape[0] > 0:
                    loss_dis = loss_fct(class_logits[kwargs['class_golds']!=-1], kwargs['class_golds'][kwargs['class_golds']!=-1])
                
            if kwargs['dis_pseudo_loss_weight'][0].item() != 0:
                if kwargs['pseudo_golds'][kwargs['pseudo_golds']!=-1].shape[0] > 0:
                    loss_dis_pseudo = loss_fct(class_logits[kwargs['pseudo_golds']!=-1], kwargs['pseudo_golds'][kwargs['pseudo_golds']!=-1])

            mlp_hidden_states = class_hidden_states
            com_hidden_states = class_hidden_states

            mlp_class_logits = self.score_dis(mlp_hidden_states)
            com_class_logits = mlp_class_logits

        elif kwargs['is_mlp'][0].item() >= 1: ## mlp
            mlp_hidden_states = self.class_mlp(class_hidden_states)
            mlp_class_logits = self.score_dis(mlp_hidden_states)
            
            assert kwargs['dis_loss_weight'][0].item() != 0
            if kwargs['class_golds'][kwargs['class_golds']!=-1].shape[0] > 0:
                loss_dis = loss_fct(mlp_class_logits[kwargs['class_golds']!=-1], kwargs['class_golds'][kwargs['class_golds']!=-1])

            if kwargs['pseudo_golds'][kwargs['pseudo_golds']!=-1].shape[0] > 0:
                loss_dis_pseudo = loss_fct(mlp_class_logits[kwargs['pseudo_golds']!=-1], kwargs['pseudo_golds'][kwargs['pseudo_golds']!=-1])

            z1, z2, com_hidden_states = self.cca_proj(class_hidden_states, mlp_hidden_states)
            com_class_logits = self.score_com(com_hidden_states)

            if self.training:
                if  (kwargs['pseudo_golds'] != -1).sum() != 0:
                    if kwargs['cca_loss_weight'][0].item() != 0 or kwargs['cca_pseudo_loss_weight'][0].item() != 0:
                        mask = (kwargs['class_golds']!=-1) | (kwargs['pseudo_golds']!=-1)
                        z1 = z1[mask]
                        z2 = z2[mask]
                        if mask.sum() > 4:
                            if kwargs['cca_loss_func'][0].item() <= 2:
                                loss_cca = self.cca_proj.calculate_loss(z1, z2, mode=kwargs['cca_loss_func'][0].item())
                            else:
                                loss_cca = self.soft_cca(z1, z2)
                    
                    # if kwargs['cca_pseudo_loss_weight'][0].item() != 0 and mlp_hidden_states is not None and (kwargs['token_type_ids']==5).sum() != 0:
                    #     mask = (kwargs['token_type_ids']==5).sum(dim=-1) == 1
                    #     label1_class_hidden_states = class_hidden_states[mask]
                    #     label2_mlp_hidden_states = mlp_hidden_states[mask]
                    #     label2_class_hidden_states = hidden_states[kwargs['token_type_ids']==5]
                    #     if mask.sum() > 0:
                    #         hyper_label2_mlp_hidden_states = self.hyper_proj(label2_mlp_hidden_states)
                    #         hyper_label2_class_hidden_states = self.hyper_proj(label2_class_hidden_states)
                    #         hyper_label1_class_hidden_states = self.hyper_proj(label1_class_hidden_states)
                    #         if hyper_label2_mlp_hidden_states.shape[0] != hyper_label2_class_hidden_states.shape[0] or hyper_label2_mlp_hidden_states.shape[0] != hyper_label1_class_hidden_states.shape[0]:
                    #             print('Error')
                    #         assert hyper_label2_mlp_hidden_states.shape[0] == hyper_label2_class_hidden_states.shape[0] and hyper_label2_mlp_hidden_states.shape[0] == hyper_label1_class_hidden_states.shape[0]
                    #         loss_triplet_ranking_loss = triplet_ranking_loss(hyper_label2_mlp_hidden_states, hyper_label2_class_hidden_states, hyper_label1_class_hidden_states, margin=0.1, distance_type='hyperbolic')

                if kwargs['com_loss_weight'][0].item() != 0:
                    if kwargs['class_golds'][kwargs['class_golds']!=-1].shape[0] > 0:
                        loss_com = loss_fct(com_class_logits[kwargs['class_golds']!=-1], kwargs['class_golds'][kwargs['class_golds']!=-1])
                    
                if kwargs['com_pseudo_loss_weight'][0].item() != 0:
                    if kwargs['pseudo_golds'][kwargs['pseudo_golds']!=-1].shape[0] > 0:
                        loss_com_pseudo = loss_fct(com_class_logits[kwargs['pseudo_golds']!=-1], kwargs['pseudo_golds'][kwargs['pseudo_golds']!=-1])


        # loss = loss_gen * kwargs['gen_loss_weight'][0].item() + loss_class * kwargs['class_loss_weight'][0].item() + loss_dis * kwargs['dis_loss_weight'][0].item() + loss_com * kwargs['com_loss_weight'][0].item() + loss_cca * kwargs['cca_loss_weight'][0].item() + loss_class_pseudo * kwargs['class_pseudo_loss_weight'][0].item() + loss_dis_pseudo * kwargs['dis_pseudo_loss_weight'][0].item() + loss_com_pseudo * kwargs['com_pseudo_loss_weight'][0].item() + loss_triplet_ranking_loss * kwargs['cca_pseudo_loss_weight'][0].item()
        loss = loss_gen * kwargs['gen_loss_weight'][0].item() + loss_class * kwargs['class_loss_weight'][0].item() + loss_dis * kwargs['dis_loss_weight'][0].item() + loss_com * kwargs['com_loss_weight'][0].item() + loss_cca * kwargs['cca_loss_weight'][0].item() + loss_class_pseudo * kwargs['class_pseudo_loss_weight'][0].item() + loss_dis_pseudo * kwargs['dis_pseudo_loss_weight'][0].item() + loss_com_pseudo * kwargs['com_pseudo_loss_weight'][0].item()

        outputs = {
            "loss": loss,
            "logits": logits,
            "mlp_class_logits": mlp_class_logits,
            "com_class_logits": com_class_logits,
            "class_hidden_states": class_hidden_states,
            "mlp_hidden_states": mlp_hidden_states,
            "com_hidden_states": com_hidden_states,
        }
        return outputs
    


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
