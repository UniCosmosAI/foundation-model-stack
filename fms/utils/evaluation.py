from typing import List, Tuple

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance  # type: ignore
from lm_eval.api.model import LM  # type: ignore
from lm_eval.api.registry import register_model  # type: ignore
from torch import nn

from fms.utils import tokenizers


@register_model("fms")
class FMSEvalHarnessLM(LM):
    """
    A wrapper class for FMS models to be used with the lm-evaluation-harness library.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: tokenizers.BaseTokenizer,
        device="cpu",
        rank=0,
        world_size=1,
    ):
        """
        Initializes the FMSEvalHarnessLM.

        Args:
            model (nn.Module): The FMS model to be evaluated.
            tokenizer (tokenizers.BaseTokenizer): The tokenizer to be used.
            device (str, optional): The device to run the model on. Defaults to "cpu".
            rank (int, optional): The rank of the current process. Defaults to 0.
            world_size (int, optional): The total number of processes. Defaults to 1.
        """
        self.wrapped_model = model
        self.tokenizer = tokenizer
        self._rank = rank
        self._world_size = world_size
        self.device = device

        # workaround for https://github.com/EleutherAI/lm-evaluation-harness/issues/1333
        # until the fix is in a release
        def generic_object():
            return None

        self.model = generic_object
        self.model.config = generic_object  # type: ignore
        self.model.config._name_or_path = "FMSEvalHarnessLM"  # type: ignore

    def loglikelihood_one(self, context: str, continuation: str) -> Tuple[float, bool]:
        """
        Calculates the log-likelihood of a single continuation given a context.

        Args:
            context (str): The context string.
            continuation (str): The continuation string.

        Returns:
            Tuple[float, bool]: A tuple containing the log-likelihood and a boolean indicating if the prediction was greedy.
        """
        context_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(context)
        )
        if not len(context_ids):
            context_ids = [self.tokenizer.bos_token_id]

        continuation_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(continuation)
        )
        input_ids = context_ids + continuation_ids[:-1]
        input_ids = torch.tensor(
            input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        logits = F.log_softmax(self.wrapped_model(input_ids)[0], -1)
        continuation_probs = logits[len(context_ids) - 1 :]
        loglikelihood = torch.gather(
            continuation_probs, 1, torch.tensor(continuation_ids).unsqueeze(1)
        ).squeeze()
        predicted = torch.argmax(continuation_probs, -1).tolist()
        greedy = predicted == continuation_ids
        return loglikelihood.sum().cpu().item(), greedy

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """
        Calculates the log-likelihood of a list of requests.

        Args:
            requests (List[Instance]): A list of requests, where each request is an instance with a context and a continuation.

        Returns:
            List[Tuple[float, bool]]: A list of tuples containing the log-likelihood and a boolean indicating if the prediction was greedy for each request.
        """
        result = []
        for request in requests:
            context, continuation = request.args
            result.append(self.loglikelihood_one(context, continuation))
        return result

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        """
        Not implemented.
        """
        raise NotImplementedError("not implemented yet")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """
        Not implemented.
        """
        raise NotImplementedError("not implemented yet")
