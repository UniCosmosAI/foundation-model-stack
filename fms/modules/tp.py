from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set, Union

import torch
import torch.nn as nn
import torch.distributed as dist


def _get_tpd_module(module: nn.Module, attr_name: str):
    """
    Get a module from a parent module by attribute name.

    Args:
        module (nn.Module): The parent module.
        attr_name (str): The name of the attribute to get.

    Returns:
        nn.Module: The requested module.
    """
    if attr_name == "self":
        return module
    return getattr(module, attr_name)


class ShardType(Enum):
    """
    The type of sharding to perform.
    """

    SHARD = 1
    RANK0 = 2
    CLONE = 3


class TPModule(nn.Module, metaclass=ABCMeta):
    """
    This is an abstract class that any nn.Module can implement to enable
    Tensor Parallel. On top of inheriting from this class, the TP module
    will have to implement list_colwise_weights, list_rowwise_weights,
    list_embedding_weights, and import_module for their relevant weights.
    Finally, the module must call setup_tp at the end of their __init__
    function. See examples in attention.py, feedforward.py and embedding.py
    """

    rank: int
    world_size: int
    group: dist.ProcessGroup

    def setup_tp(self, rank: int, group: Optional[dist.ProcessGroup]) -> None:
        """
        Set up the tensor parallel module.

        Args:
            rank (int): The rank of the current process.
            group (Optional[dist.ProcessGroup]): The process group for tensor parallelism.
        """
        self.rank = rank
        if group is not None:
            self.group = group
        elif dist.group.WORLD is not None:
            self.group = dist.group.WORLD
        else:
            raise ValueError("No process group defined!")
        self.world_size = self.group.size()

    def __get_tp_slices(
        self,
        input_size,
        output_size_per_partition,
        max_partition_sizes,
    ):
        """
        Get the tensor parallel slices for a given input size.

        Args:
            input_size (int): The size of the input tensor.
            output_size_per_partition (int): The size of the output tensor per partition.
            max_partition_sizes (List[int]): The maximum partition sizes.

        Yields:
            slice: The tensor parallel slice.
        """
        cusum_max_partition_sizes = [0]
        min_partition_size = min(max_partition_sizes)
        for m in max_partition_sizes:
            cusum_max_partition_sizes.append(
                cusum_max_partition_sizes[-1] + (m // min_partition_size)
            )

        for weight_i, replication in enumerate(max_partition_sizes):
            # The shard to copy based on the process rank and the replication threshold
            sharding_rank = self.rank // max(1, self.world_size // replication)
            # The number of elements to shard out of the tensor
            tensor_shard_size = output_size_per_partition * (
                max(self.world_size, replication)
                // max(self.world_size, min_partition_size)
            )
            # For fused weights, where to start extracting the shard
            tensor_shard_offset = (
                cusum_max_partition_sizes[weight_i]
                * input_size
                // cusum_max_partition_sizes[-1]
            )
            yield slice(
                sharding_rank * tensor_shard_size + tensor_shard_offset,
                (sharding_rank + 1) * tensor_shard_size + tensor_shard_offset,
            )

    def sharded_copy(
        self,
        param: Union[torch.nn.Parameter, torch.Tensor],
        tensor_value: torch.Tensor,
        dim: int,
        max_partition_sizes: List[int],
        shard_type: ShardType = ShardType.SHARD,
    ):
        """
        This function copies the correct shard of the weights for a rowwise-TP'd module
        according to the rank of the process and the world_size.

        Args:
            param (Union[torch.nn.Parameter, torch.Tensor]): Parameter that has had TP applied.
            tensor_value (torch.Tensor): Tensor that needs sharding.
            dim (int): Dimension on which to shard. colwise sharding is usually dim 0, rowwise is usually dim 1.
            max_partition_sizes (List[int]): For each number in the list, if world_size is smaller than or equal to that number, the tensor will get partitioned in worldsize parts, else if world size is larger than the number then you will get world size parts replicated in worldsize / number.
            shard_type (ShardType): The type of sharding to perform.
        """
        if shard_type == ShardType.SHARD:
            # In the case where world size is larger than any of the partition sizes, we must add replication up til the
            # world size per partition
            max_partition_sizes_replicated = [
                max(self.world_size, m) for m in max_partition_sizes
            ]
            # Divide the weight matrix along the second dimension.
            output_size_per_partition = param.shape[dim] // (
                sum(max_partition_sizes_replicated)
                // min(max_partition_sizes_replicated)
            )
            tp_slices = self.__get_tp_slices(
                tensor_value.shape[dim], output_size_per_partition, max_partition_sizes
            )
            tp_shard_indices = [
                tuple([slice(None) for _ in range(dim)] + [tp_slice])
                for tp_slice in tp_slices
            ]
            tensors_to_cat = [
                tensor_value[tp_shard_index] for tp_shard_index in tp_shard_indices
            ]
            tensor = torch.cat(tensors_to_cat, dim=dim)
            param.copy_(tensor, non_blocking=True)
        elif shard_type == ShardType.RANK0:
            if self.rank == 0:
                param.copy_(tensor_value, non_blocking=True)
            else:
                param.zero_()
        else:  # ShardType.CLONE
            param.copy_(tensor_value, non_blocking=True)

    def _get_sd_weight(
        self,
        state_dict: Dict[str, torch.Tensor],
        used_keys: Set[str],
        substr_matches: List[str],
    ):
        """
        Extract from partial model state_dict, the tensor
        uniquely identified by the matching substrings provided.

        Args:
            state_dict (Dict[str, torch.Tensor]): The state dictionary.
            used_keys (Set[str]): A set of used keys.
            substr_matches (List[str]): A list of substrings to match.

        Returns:
            torch.Tensor: The extracted tensor.
        """
        results = []
        for k in state_dict:
            all_matches = True
            for substr_match in substr_matches:
                if substr_match not in k.split("."):
                    all_matches = False
            if all_matches:
                used_keys.add(k)
                results.append(state_dict[k])
        if len(results) == 1:
            return results[0]
        elif len(results) > 1:
            raise ValueError(
                f"Conditions not stringent enough: keys are {', '.join(state_dict.keys())}"
            )
        else:
            raise ValueError(
                f"Weight not found, searching for {substr_matches} "
                f"but weights names are {', '.join(state_dict.keys())}"
            )

    def load_weights(
        self,
        tensor_values: Dict[str, torch.Tensor],
    ):
        """
        Load all tensor values into a TP module.

        Override this method to load weights into a TP module. You can see
        examples for all TP modules in FMS, but the functions will generally
        have the following structure:

        1. Grab the necessary weights from tensor_values. Which weights
            these are depend on each module.
        2. Raise exceptions for missing required weights in tensor_values
            (ValueError) or for unused weights in tensor_values
            (AttributeError)
        3. Use the sharded_copy method to load and shard each weight correctly
            into the TP module. Pay special attention to the max_partition_sizes
            parameter for supporting fused weights and MQA/GQA among others.

        PSA: Each weight has a List[int] (max_partition_sizes) associated
        with it where if the list is larger than size 1, this implies the
        weight is fused (where the length of the list is the number of fused
        weights in a single parameter).

        Args:
            tensor_values (Dict[str, torch.Tensor]): a state dict containing all the weights for a TP module.
        """
        pass

    @staticmethod
    @abstractmethod
    def import_module(module, group: dist.ProcessGroup):
        """
        Import a module to a TP module.

        Args:
            module (nn.Module): The module to import.
            group (dist.ProcessGroup): The process group for tensor parallelism.
        """
        pass
