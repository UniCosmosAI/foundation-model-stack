import abc
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, _Loss
from transformers import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    Seq2SeqLMOutput,
    SequenceClassifierOutput,
)
from transformers.utils import ModelOutput

from fms.modules.head import MLPClassificationHead
from fms.utils.activation import str_to_activation


class LMHeadMixin:
    """
    Base class to represent enabled a model architecture to include the lm_head. An lm_head can either be given by the
    child class, or created by the LMHeadMixin. In either case, both lm_head should be the same type resulting in the
    same state dict.

    This mixin is responsible for:
        - creating and holding the lm_head nn.Module
        - computation of the loss
        - producing the proper lm_head output dataclass
    """

    def __init__(
        self,
        config: PretrainedConfig,
        lm_head: Optional[nn.Module] = None,
        _lm_head_params: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize an LMHeadMixin

        Args:
            config (PretrainedConfig): The model config.
            lm_head (Optional[nn.Module]): If given, the lm_head will simply be set in the underlying architecture.
                If not given, the lm_head will be created by this class.
            _lm_head_params (Optional[Dict[str, Any]]): An optional dictionary of parameters that do not have a standard
                name in the config, but are required for creation of the empty lm_head.
            *args: Passthrough for the mixin.
            **kwargs: Passthrough for the mixin.
        """
        self.config = config
        # if lm head was already given, we do not need to create it, otherwise we create a fresh lm_head
        # lm_head is not None when a subclass of HFModelArchitecture provides a model with an lm_head
        if lm_head is None:
            lm_head = self._get_empty_lm_head(
                **({} if _lm_head_params is None else _lm_head_params)
            )

        super().__init__(config=config, lm_head=lm_head, *args, **kwargs)

    @abc.abstractmethod
    def _get_empty_lm_head(self, **kwargs) -> nn.Module:
        """
        Get an empty initialized lm_head given specific parameters provided by a child implementation of this class.

        Args:
            **kwargs: If _lm_head_params dict is given in __init__, the child class implementing this method will
                include those parameters ONLY in these kwargs.

        Returns:
            nn.Module: The empty lm head module.
        """
        pass

    @abc.abstractmethod
    def _compute_loss(self, prediction: torch.Tensor, labels: torch.Tensor) -> _Loss:
        """
        Compute the loss between predictions and labels.

        Args:
            prediction (torch.Tensor): Prediction from a forward step of a module.
            labels (torch.Tensor): The labels to compare.

        Returns:
            _Loss: The loss object computed from the prediction/labels.
        """
        pass

    @abc.abstractmethod
    def _lm_head(
        self,
        input_ids: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Adapt your given pytorch native lm head to that of the one expected in huggingface. Note: This is not
        required if your lm_head simply takes in the input_ids and returns a torch.Tensor.

        Args:
            input_ids (torch.Tensor): The downstream input_ids (either from other forward passes or a dataset).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The output from the forward function of the lm_head if an lm_head exists.
        """
        return self.lm_head(input_ids)

    @abc.abstractmethod
    def _produce_lm_output(
        self,
        logits: torch.FloatTensor,
        loss: _Loss,
        encoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions],
        decoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions],
    ) -> ModelOutput:
        """
        Produce the proper lm head output dataclass given the output from the encoder, decoder, loss, and lm head.

        Args:
            logits (torch.FloatTensor): The output logits from the lm head.
            loss (_Loss): The loss object returned from _compute_loss.
            encoder_outputs (Optional[BaseModelOutputWithPastAndCrossAttentions]): The output from the encoder.
            decoder_outputs (Optional[BaseModelOutputWithPastAndCrossAttentions]): The output from the decoder.

        Returns:
            ModelOutput: A ModelOutput object based on the task of the lm head.
        """
        pass

    def get_output_embeddings(self):
        """
        Get the output embeddings from the lm_head.

        Returns:
            nn.Module: The lm_head module.
        """
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """
        Set the output embeddings of the lm_head.

        Args:
            new_embeddings (nn.Module): The new lm_head module.
        """
        self.lm_head = new_embeddings


class LMHeadModelLMHeadMixin(LMHeadMixin):
    """Provides a decoder model with a `language modeling` head"""

    def __init__(self, bias: bool, *args, **kwargs):
        """
        Initialize a LMHeadModelLMHeadMixin.

        Args:
            bias (bool): Vocab bias used in creating the lm_head.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(_lm_head_params={"bias": bias}, *args, **kwargs)

    def _get_empty_lm_head(self, bias: bool) -> nn.Module:
        return nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=bias)

    def _compute_loss(self, prediction: torch.Tensor, labels: torch.Tensor) -> _Loss:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        shift_output = prediction[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return loss_fn(
            shift_output.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
        )

    def _produce_lm_output(
        self,
        logits: torch.FloatTensor,
        loss: _Loss,
        encoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions],
        decoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions],
    ) -> ModelOutput:
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )


class ConditionalGenerationLMHeadMixin(LMHeadMixin):
    """Provides an encoder-decoder model with a `language modeling` head"""

    def __init__(self, bias: bool, *args, **kwargs):
        """
        Initialize a ConditionalGenerationLMHeadMixin.

        Args:
            bias (bool): Vocab bias used in creating the lm_head.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(_lm_head_params={"bias": bias}, *args, **kwargs)

    def _get_empty_lm_head(self, bias: bool) -> nn.Module:
        return nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=bias)

    def _compute_loss(self, prediction: torch.Tensor, labels: torch.Tensor) -> _Loss:
        loss_fn = nn.CrossEntropyLoss()
        inds = labels.view(-1).sub(self.config.pad_token_id).nonzero().squeeze(1)
        loss = loss_fn(
            prediction.view(-1, self.config.vocab_size)[inds],
            labels.view(-1)[inds],
        )
        return loss

    def _produce_lm_output(
        self,
        logits: torch.FloatTensor,
        loss: _Loss,
        encoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions],
        decoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions],
    ) -> ModelOutput:
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class SequenceClassificationLMHeadMixin(LMHeadMixin):
    """
    Provides a model architecture with a sequence classification/regression head

    Depending on the problem type, this class will provide different functionality. Problem types can be one of
    "regression", "single_label_classification", "multi_label_classification", None. If None, problem_type will be
    set at run-time based on config.num_labels and the label dtype.
    """

    _tied_weights_keys = [
        "lm_head.head.weight",
        "lm_head.head.bias",
    ]

    def __init__(
        self,
        classifier_activation_fn: str = "tanh",
        classifier_dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        """
        Initialize a SequenceClassificationLMHeadMixin.

        Args:
            classifier_activation_fn (str): The activation function name to use in creating the lm_head.
                Will be ignored if depth is 0. (default is tanh)
            classifier_dropout (float): The dropout to be used in the lm head (default is 0.1).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

        super().__init__(
            _lm_head_params={
                "classifier_activation_fn": classifier_activation_fn,
                "classifier_dropout": classifier_dropout,
            },
            *args,
            **kwargs,
        )

    def _get_empty_lm_head(
        self,
        classifier_activation_fn: str,
        classifier_dropout: float,
    ) -> nn.Module:
        return MLPClassificationHead(
            self.config.hidden_size,
            self.config.num_labels,
            str_to_activation(classifier_activation_fn),
            dropout=classifier_dropout,
            apply_pooling_fn=True,
        )

    def _compute_loss(self, prediction: torch.Tensor, labels: torch.Tensor) -> _Loss:
        if self.config.problem_type is None:
            if self.config.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.config.num_labels > 1 and (
                labels.dtype == torch.long or labels.dtype == torch.int
            ):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.config.num_labels == 1:
                loss = loss_fct(prediction.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(prediction, labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                prediction.view(-1, self.config.num_labels), labels.view(-1)
            )
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(prediction, labels)

        return loss

    def _produce_lm_output(
        self,
        logits: torch.FloatTensor,
        loss: _Loss,
        encoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions],
        decoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions],
    ) -> ModelOutput:
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_output_embeddings(self):
        """
        Get the output embeddings from the lm_head's head.

        Returns:
            nn.Module: The head of the lm_head module.
        """
        return self.lm_head.head


class MaskedLMHeadMixin(LMHeadMixin):
    """Provides a model architecture with a masked lm head"""

    _tied_weights_keys = [
        "lm_head.head.weight",
        "lm_head.head.bias",
    ]

    def __init__(
        self,
        activation_fn: str = "gelu",
        norm_eps: float = 1e-12,
        *args,
        **kwargs,
    ):
        """
        Initialize a MaskedLMHeadMixin.

        Args:
            activation_fn (str): The activation function name to use in creating the lm_head.
                Will be ignored if depth is 0. (default is gelu)
            norm_eps (float): Norm epsilon for the model.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            _lm_head_params={
                "activation_fn": activation_fn,
                "norm_eps": norm_eps,
            },
            *args,
            **kwargs,
        )

    def get_output_embeddings(self):
        """
        Get the output embeddings from the lm_head's head.

        Returns:
            nn.Module: The head of the lm_head module.
        """
        return self.lm_head.head

    def _get_empty_lm_head(self, activation_fn: str, norm_eps: float) -> nn.Module:
        return MLPClassificationHead(
            self.config.hidden_size,
            self.config.vocab_size,
            activation_fn=str_to_activation(activation_fn),
            layer_norm=nn.LayerNorm(self.config.hidden_size, norm_eps),
        )

    def _compute_loss(self, prediction: torch.Tensor, labels: torch.Tensor) -> _Loss:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fn(prediction.view(-1, self.config.vocab_size), labels.view(-1))

    def _produce_lm_output(
        self,
        logits: torch.FloatTensor,
        loss: _Loss,
        encoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions],
        decoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions],
    ) -> ModelOutput:
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
