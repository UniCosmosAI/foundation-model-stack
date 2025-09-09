import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple
from typing_extensions import Unpack

import torch
import torch.nn as nn

from fms import models
from fms.distributed.strategy import DistributedStrategy, NoOpStrategy
from fms.modules.attention import (
    AttentionKwargs,
    MultiHeadAttention,
    get_attention_type,
)
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.linear import get_linear_type
from fms.modules.positions import RotaryEmbedding
from fms.utils import serialization
from fms.utils.activation import str_to_activation
from fms.utils.config import ModelConfig


logger = logging.getLogger(__name__)


@dataclass
class GraniteConfig(ModelConfig):
    """
    Configuration for the Granite model.

    Args:
        src_vocab_size (int): The size of the source vocabulary.
        emb_dim (int): The embedding dimension.
        norm_eps (float): The epsilon value for layer normalization.
        nheads (int): The number of attention heads.
        kvheads (int): The number of key-value heads.
        nlayers (int): The number of layers in the model.
        pad_id (int): The ID of the padding token.
        hidden_grow_factor (float): The growth factor for the hidden layer in the feed-forward network.
        multiple_of (int): The multiple of value for the feed-forward network.
        activation_fn (str): The activation function to use.
        p_dropout (float): The dropout probability.
        max_expected_seq_len (int): The maximum expected sequence length.
        ntk_scaling (bool): Whether to use NTK scaling for RoPE.
        attn_bias (bool): Whether to use bias in the attention layer.
        mlp_bias (bool): Whether to use bias in the MLP layer.
        tie_heads (bool): Whether to tie the embedding and output heads.
        rope_theta (float): The theta value for RoPE.
        embedding_multiplier (float): The multiplier for the embedding layer.
        logits_scaling (float): The scaling factor for the logits.
        residual_multiplier (float): The multiplier for the residual connection.
        attention_multiplier (float): The multiplier for the attention scores.
        linear_config (Optional[Mapping[str, Any]]): The configuration for the linear layers.
        fused_weights (bool): Whether to use fused weights.
    """

    src_vocab_size: int = 32_000  # can be set by tokenizer
    emb_dim: int = 4096
    norm_eps: float = 1e-5
    nheads: int = 32
    kvheads: int = 0
    nlayers: int = 32
    pad_id: int = -1
    hidden_grow_factor: float = 8 / 3
    multiple_of: int = 256
    activation_fn: str = "swish"
    p_dropout: float = 0.0
    max_expected_seq_len: int = 4096
    ntk_scaling: bool = False
    attn_bias: bool = False
    mlp_bias: bool = False
    tie_heads: bool = False
    rope_theta: float = 10_000.0
    embedding_multiplier: float = 1.0
    logits_scaling: float = 1.0
    residual_multiplier: float = 1.0
    attention_multiplier: float = 1.0
    linear_config: Optional[Mapping[str, Any]] = None
    fused_weights: bool = True


class GraniteBlock(nn.Module):
    """
    A single block of the Granite model.

    Args:
        config (GraniteConfig): The configuration for the Granite model.
        rotary_emb (RotaryEmbedding): The rotary embedding layer.
    """

    def __init__(self, config: GraniteConfig, rotary_emb: RotaryEmbedding):
        super(GraniteBlock, self).__init__()
        self.config = config
        emb_kq = self.config.emb_dim // self.config.nheads
        emb_v = self.config.emb_dim // self.config.nheads

        self.ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_ln = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )

        if self.config.kvheads == 0:
            kvheads = self.config.nheads
        else:
            kvheads = self.config.kvheads
            assert self.config.nheads % self.config.kvheads == 0

        self.attn = MultiHeadAttention(
            self.config.emb_dim,
            emb_kq,
            emb_v,
            self.config.nheads,
            kvheads,
            p_dropout=self.config.p_dropout,
            use_bias=self.config.attn_bias,
            position_encoder=rotary_emb,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
            scale_factor=self.config.attention_multiplier,
        )
        self.ff_sub_layer = GatedLinearUnit(
            self.config.emb_dim,
            hidden_grow_factor=self.config.hidden_grow_factor,
            multiple_of=self.config.multiple_of,
            activation_fn=str_to_activation(self.config.activation_fn),
            p_dropout=self.config.p_dropout,
            use_bias=self.config.mlp_bias,
            fused=self.config.fused_weights,
            linear_config=self.config.linear_config,
        )

        if self.config.p_dropout != 0:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def forward(
        self,
        x,
        *,
        position_ids=None,
        past_key_value_state=None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        """
        Forward pass for the GraniteBlock.

        Args:
            x (torch.Tensor): The input tensor.
            position_ids (Optional[torch.LongTensor]): The position IDs for the input tensor.
            past_key_value_state (Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]): The past key-value state for caching.
            use_cache (bool): Whether to use caching.
            **attn_kwargs (Unpack[AttentionKwargs]): Additional keyword arguments for the attention layer.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.FloatTensor, torch.FloatTensor]]]:
                The output tensor, and the new cache if use_cache is True.
        """
        # if the cache is not empty, we need to get the kv cache for self and cross attention
        self_attn_past_key_value = past_key_value_state

        # first we do MHA and Add&Norm
        residual = x
        x = self.ln(x)
        x = self.attn(
            q=x,
            position_ids=position_ids,
            past_key_value_state=self_attn_past_key_value,
            use_cache=use_cache,
            **attn_kwargs,
        )
        cache = None
        if use_cache:
            x, cache = x
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # residual connection
        x = x * self.config.residual_multiplier + residual

        # then we do FF and Add&Norm
        residual = x
        x = self.ff_ln(x)
        x = self.ff_sub_layer(x)
        if self.config.p_dropout != 0:
            x = self.dropout(x)
        # another residual
        x = x * self.config.residual_multiplier + residual

        if use_cache:
            return (x, cache)
        else:
            return x


class GraniteHeadless(nn.Module):
    """
    The Granite model without the language model head.

    Args:
        config (Optional[GraniteConfig]): The configuration for the Granite model.
        distributed_strategy (DistributedStrategy): The distributed strategy to use.
        **kwargs: Additional keyword arguments to update the configuration.
    """

    def __init__(
        self,
        config: Optional[GraniteConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(GraniteHeadless, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = GraniteConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.width = self.config.emb_dim
        self.pad_id = self.config.pad_id
        self.max_expected_seq_len = self.config.max_expected_seq_len

        self.embedding = nn.Embedding(
            self.config.src_vocab_size,
            self.config.emb_dim,
            padding_idx=self.config.pad_id,
        )

        rope_scaling = {"rope_type": "ntk" if self.config.ntk_scaling else "regular"}

        self.rot_emb = RotaryEmbedding(
            dim=self.config.emb_dim // self.config.nheads,
            scaling=rope_scaling,
            max_seq_len=self.config.max_expected_seq_len,
            ratio=self.config.rope_theta,
        )
        # RoPE init
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        layers = []
        for i in range(self.config.nlayers):
            block: nn.Module = GraniteBlock(self.config, self.rot_emb)
            block = self.distributed_strategy.distribute_layer(block, i)
            layers.append(block)
        self.layers = nn.ModuleList(layers)

        dec_norm = LayerNormParameterized(
            self.config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=self.config.norm_eps,
            use_high_precision_pow=True,
        )
        self.dec_norm = self.distributed_strategy.distribute_module(
            dec_norm, final_layers=True
        )

        if self.config.p_dropout:
            self.dropout = nn.Dropout(self.config.p_dropout)

    def reset_parameters(self):
        """
        Resets the parameters of the model.
        """
        nn.init.trunc_normal_(
            self.embedding.weight, mean=0.0, std=self.config.emb_dim**-0.5
        )

        # RoPE init
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

        # Call reset_parameters for relevant sub-layers
        for m in self.modules():
            if (
                isinstance(m, MultiHeadAttention)
                or isinstance(m, GatedLinearUnit)
                or isinstance(m, LayerNormParameterized)
            ):
                m.reset_parameters()

    def _clean_up_rot_emb_cache(
        self,
        cached_freqs: dict[Optional[torch.device], dict[int, torch.Tensor]],
        max_seq_len_cached: dict[Optional[torch.device], int],
    ):
        """
        Cleans up the rotary embedding cache by removing meta tensors.

        Args:
            cached_freqs (dict): The cached frequencies.
            max_seq_len_cached (dict): The maximum sequence length cached.
        """
        # remove meta tensors from cached_freqs
        for dev in list(cached_freqs.keys()):
            for alp in list(cached_freqs[dev].keys()):
                if cached_freqs[dev][alp].device == torch.device("meta"):
                    del cached_freqs[dev][alp]
                    if len(cached_freqs[dev]) == 0:
                        del cached_freqs[dev]
                        del max_seq_len_cached[dev]

    def post_init(self):
        """
        Performs post-initialization steps, such as cleaning up the rotary embedding cache and initializing RoPE on the correct device.
        """
        # This function is called in `get_model` after the model is
        # fully initalized on the correct device

        self._clean_up_rot_emb_cache(
            self.rot_emb.cached_freqs,
            self.rot_emb.max_seq_len_cached,
        )

        # init RoPE on the right device(s)
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)

    def forward(
        self,
        x_in,
        position_ids=None,
        past_key_value_states=None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        """
        Forward pass for the GraniteHeadless model.

        Args:
            x_in (torch.Tensor): The input tensor.
            position_ids (Optional[torch.LongTensor]): The position IDs for the input tensor.
            past_key_value_states (Optional[List[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]]):
                The past key-value states for caching.
            use_cache (bool): Whether to use caching.
            **attn_kwargs (Unpack[AttentionKwargs]): Additional keyword arguments for the attention layer.

        Returns:
            Tuple[torch.Tensor, List[Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]]:
                The output tensor and the new cache if use_cache is True.
        """
        # Embed the given vocabulary indices using the given attention mask, with pre-/post-norm and dropout as specified
        # x_in: batch_size x seq_len x emb_dim if input is already embedded, otherwise batch_size x seq_len
        # mask: batch_size x seq_len x seq_len
        # bias: nheads x seq_len x seq_len
        if past_key_value_states is None or len(past_key_value_states) == 0:
            past_key_value_states = [None for _ in range(len(self.layers))]

        if x_in.dim() == 2:  # input is not already embedded
            x_in = self.embedding(x_in)
        x_in = x_in * self.config.embedding_multiplier

        # this is the output cache for all the decoder layers
        present_key_value_states = []

        for i, layer in enumerate(self.layers):
            output = layer(
                x=x_in,
                position_ids=position_ids,
                past_key_value_state=past_key_value_states[i],
                use_cache=use_cache,
                **attn_kwargs,
            )

            if use_cache:
                x_in, present_key_value_state = output
                present_key_value_states.append(present_key_value_state)

            else:
                x_in = output

        dec_out = x_in
        dec_out = self.dec_norm(dec_out)
        if self.config.p_dropout:
            dec_out = self.dropout(dec_out)

        return dec_out, present_key_value_states


class Granite(nn.Module):
    """
    The Granite model.

    Args:
        config (Optional[GraniteConfig]): The configuration for the Granite model.
        distributed_strategy (DistributedStrategy): The distributed strategy to use.
        **kwargs: Additional keyword arguments to update the configuration.
    """

    def __init__(
        self,
        config: Optional[GraniteConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(Granite, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = GraniteConfig()
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.base_model = GraniteHeadless(self.config, self.distributed_strategy)
        self.head = nn.Linear(
            self.config.emb_dim, self.config.src_vocab_size, bias=False
        )

    @classmethod
    def from_config(cls, config: GraniteConfig) -> "Granite":
        """
        Creates a Granite model from a configuration object.

        Args:
            config (GraniteConfig): The configuration for the Granite model.

        Returns:
            Granite: The Granite model.
        """
        return cls(config)

    def get_config(self) -> GraniteConfig:
        """
        Returns the configuration of the model.

        Returns:
            GraniteConfig: The configuration of the model.
        """
        return self.config

    def reset_parameters(self):
        """
        Resets the parameters of the model.
        """
        self.head.weight.data.normal_(
            0,
            1 / math.sqrt(math.sqrt(self.config.emb_dim * self.config.src_vocab_size)),
        )
        self.base_model.reset_parameters()

    def post_init(self):
        """
        Performs post-initialization steps, such as tying the embedding and output heads.
        """
        # if this model ties weights, they are tied here
        if self.config.tie_heads:
            # handle assignment of non-meta weights to meta parameters
            if self.head.weight.device == torch.device("meta"):
                self.head.weight = self.base_model.embedding.weight
            else:
                self.base_model.embedding.weight = self.head.weight

        self.base_model.post_init()

    def forward(
        self,
        x: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value_states: Optional[Tuple[torch.FloatTensor,]] = None,
        use_cache: bool = False,
        only_last_token: bool = False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        """
        Forward pass for the Granite model.

        Args:
            x (torch.LongTensor): The input tensor.
            position_ids (Optional[torch.LongTensor]): The position IDs for the input tensor.
            past_key_value_states (Optional[Tuple[torch.FloatTensor,]]): The past key-value states for caching.
            use_cache (bool): Whether to use caching.
            only_last_token (bool): Whether to only return the predictions for the last token.
            **attn_kwargs (Unpack[AttentionKwargs]): Additional keyword arguments for the attention layer.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.FloatTensor,]]]:
                The output predictions, and the new cache if use_cache is True.
        """
        get_attention_type(**attn_kwargs)["validate_attn_kwargs"](
            input_ids=x,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            **attn_kwargs,
        )

        output, cache = self.base_model(
            x,
            position_ids,
            past_key_value_states,
            use_cache,
            **attn_kwargs,
        )

        if only_last_token:
            output = output[:, -1, :]
        preds = self.head(output)
        preds = preds / self.config.logits_scaling

        if use_cache:
            return preds, cache
        else:
            return preds


_8b_config = GraniteConfig(
    src_vocab_size=49155,
    emb_dim=4096,
    norm_eps=1e-5,
    nheads=32,
    kvheads=8,
    nlayers=40,
    hidden_grow_factor=12800 / 4096,
    max_expected_seq_len=8192,
    rope_theta=10_000.0,
    pad_id=0,
    p_dropout=0.0,  # overwriting config.json
    tie_heads=True,
    embedding_multiplier=12.0,
    logits_scaling=16.0,
    residual_multiplier=0.22,
    attention_multiplier=0.0078125,
)

_3_1_2b_config = GraniteConfig(
    src_vocab_size=49155,
    emb_dim=2048,
    norm_eps=1e-5,
    nheads=32,
    kvheads=8,
    nlayers=40,
    hidden_grow_factor=8192 / 2048,
    max_expected_seq_len=131072,
    rope_theta=5000000.0,
    pad_id=0,
    p_dropout=0.0,
    tie_heads=True,
    embedding_multiplier=12.0,
    logits_scaling=8.0,
    residual_multiplier=0.22,
    attention_multiplier=0.015625,
)

_architecture_name = "granite"


def _granite_factory_factory(config):
    """
    A factory function that creates a factory function for a Granite model with a given configuration.

    Args:
        config (GraniteConfig): The configuration for the Granite model.

    Returns:
        Callable: A factory function that creates a Granite model.
    """

    def factory(**kwargs):
        return Granite(config, **kwargs)

    return factory


models.register_model(_architecture_name, "8b", _granite_factory_factory(_8b_config))
models.register_model(
    _architecture_name, "3_1_2b", _granite_factory_factory(_3_1_2b_config)
)


def _weight_fusion(
    input_sd: Mapping, model_config: Optional[GraniteConfig] = None, **kwargs
):
    """
    Performs weight fusion on the state dictionary.

    Args:
        input_sd (Mapping): The input state dictionary.
        model_config (Optional[GraniteConfig]): The model configuration.
        **kwargs: Additional keyword arguments.

    Returns:
        Mapping: The modified state dictionary.
    """
    has_fused_weights = True
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False

    new_sd = input_sd
    if has_fused_weights:
        new_sd = serialization._mlp_glu_unfused_to_fused_adapter_step(
            serialization._attn_unfused_to_fused_step(new_sd)
        )
    return new_sd


serialization.register_adapter_step(_architecture_name, "weight_fusion", _weight_fusion)


def _hf_to_fms_names(input_sd: Mapping[str, Any], **kwargs) -> Mapping[str, Any]:
    """
    Converts Hugging Face model state dictionary names to FMS model state dictionary names.

    Args:
        input_sd (Mapping[str, Any]): The input state dictionary.
        **kwargs: Additional keyword arguments.

    Returns:
        Mapping[str, Any]: The converted state dictionary.
    """
    replacements = [
        (r"^lm_head.weight", "head.weight"),
        (r"^model.embed_tokens.weight", "base_model.embedding.weight"),
        (r"^model.norm", "base_model.dec_norm"),
        (r"^model.layers", "base_model.layers"),
        (r"self_attn\.k_proj", "attn.in_proj.key"),
        (r"self_attn\.v_proj", "attn.in_proj.value"),
        (r"self_attn\.q_proj", "attn.in_proj.query"),
        (r"self_attn\.o_proj", "attn.dense"),
        (r"mlp\.gate_proj", "ff_sub_layer.wg"),
        (r"mlp\.up_proj", "ff_sub_layer.w1"),
        (r"mlp\.down_proj", "ff_sub_layer.w2"),
        (r"input_layernorm", "ln"),
        (r"post_attention_layernorm", "ff_ln"),
    ]
    new_sd = {}
    for name, param in input_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param
    return new_sd


serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_names", _hf_to_fms_names
)


def _get_rope_params(linear_type: str) -> list[str]:
    """
    Returns the list of RoPE parameters for a given linear layer type.

    Args:
        linear_type (str): The type of linear layer.

    Returns:
        list[str]: The list of RoPE parameters.
    """
    if "gptq" in linear_type:
        return ["qweight", "scales", "qzeros", "bias"]
    elif "int8" in linear_type:
        # quantize_weight is fms-model-optimizer identifier of weight clip values
        return ["weight", "bias", "quantize_weight"]
    elif "fp8" in linear_type:
        return ["weight", "weight_scale", "input_scale", "bias"]
    else:  # torch.nn.Linear
        return ["weight", "bias"]


def _hf_to_fms_rope(
    input_sd: Mapping[str, Any], model_config: Optional[GraniteConfig] = None, **kwargs
) -> Mapping[str, Any]:
    """
    Converts Hugging Face RoPE parameters to FMS RoPE parameters.

    Args:
        input_sd (Mapping[str, Any]): The input state dictionary.
        model_config (Optional[GraniteConfig]): The model configuration.
        **kwargs: Additional keyword arguments.

    Returns:
        Mapping[str, Any]: The converted state dictionary.
    """
    new_sd = {}

    if model_config:
        head_size = model_config.emb_dim // model_config.nheads
    else:
        logger.warning("Missing model_config, assuming defaults for head_size")
        head_size = 128  # Good default for most models

    for name, param in input_sd.items():
        # Some checkpoints have weights in different precisions, which can have
        # auxiliary tensors (see _get_rope_params e.g. gptq, fp8).
        # Thus, we need to get rope_params per parameter.
        linear_type_str = "torch_linear"
        if model_config and model_config.linear_config:
            linear_type_str = get_linear_type(
                model_config.linear_config,
                module_name=name,
            )
        rope_params = _get_rope_params(linear_type_str)
        trans_required_pattern = re.compile(
            f"base_model.layers.[0-9]+.attn.in_proj.(query|key).({'|'.join(rope_params)})$"
        )

        # hf -> fms requires a transpose operation for the query and key
        # weight and bias parameters for Llama models
        # This transpose is due to the different implementation of RoPE in
        # HF and FMS. While FMS follows the original RoPE paper
        # (https://arxiv.org/abs/2104.09864), HF has its own implementation
        # that doesn't respect the order of outputs. This is OK as long as you
        # rearrange the weights of the query and key projections, as the
        # combination projection + RoPE ends up producing the same outputs.
        # Therefore, to make FMS produce the correct order of outputs when
        # loading from an HF checkpoint, we need to undo the transformation
        # that HF does from the original Meta weights
        is_gptq_2d_qparam = "gptq" in linear_type_str and param.dim() == 2
        if bool(trans_required_pattern.match(name)) and param.numel() > 1:
            temp = param
            if is_gptq_2d_qparam:
                # GPTQ qweights are [in_feat, out_feat] (unlike usual [out_feat, in_feat])
                # and are fully transposed before & after process.
                # GPTQ scales and qzeros are also transposed accordingly
                temp = temp.transpose(0, 1)
            # num_heads is used in the transformation required for hf->fms
            # can't be precomputed because q and k might have different num_heads
            num_heads = temp.size(0) // head_size

            if temp.dim() == 2:  # weight
                temp_view = temp.view(num_heads, 2, -1, temp.size(1))
            else:  # 1-dim parameters
                temp_view = temp.view(num_heads, 2, -1)
            temp = temp_view.transpose(1, 2).reshape(*temp.size())

            if is_gptq_2d_qparam:
                temp = temp.transpose(0, 1)

            new_sd[name] = temp
        else:
            new_sd[name] = param

    return new_sd


def _hf_gptq_granite_check(
    input_sd: Mapping[str, Any], model_config: Optional[GraniteConfig] = None, **kwargs
) -> Mapping[str, Any]:
    """
    Checks if a GPTQ Hugging Face Granite checkpoint can be loaded into a model with fused weights.

    Args:
        input_sd (Mapping[str, Any]): The input state dictionary.
        model_config (Optional[GraniteConfig]): The model configuration.
        **kwargs: Additional keyword arguments.

    Returns:
        Mapping[str, Any]: The input state dictionary.

    Raises:
        ValueError: If a GPTQ HF Granite checkpoint is being loaded into a model with fused weights.
    """
    has_fused_weights = True
    linear_type = "torch_linear"
    if model_config:
        if not model_config.fused_weights:
            has_fused_weights = False
        if model_config.linear_config:
            linear_type = model_config.linear_config["linear_type"]

    if not callable(linear_type) and "gptq" in linear_type and has_fused_weights:
        raise ValueError(
            "GPTQ HF granite checkpoints cannot be loaded into a model with fused weights"
        )

    return input_sd


serialization.register_adapter_step(
    "granite", "hf_gptq_fusion_check", _hf_gptq_granite_check
)

serialization.register_adapter_step(
    _architecture_name, "hf_to_fms_rope", _hf_to_fms_rope
)

serialization.register_adapter(
    _architecture_name,
    "hf",
    ["hf_to_fms_names", "hf_to_fms_rope", "hf_gptq_fusion_check", "weight_fusion"],
)
