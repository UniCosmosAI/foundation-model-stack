from typing import Any, List, Optional

from torch.utils.data import Dataset, IterableDataset


def _state_dict_save_helper(target):
    """
    A helper function to save the state of a dataset.

    Args:
        target: The object to save the state of.

    Returns:
        dict: The state dictionary.
    """
    if isinstance(target, dict):
        dict_attrs = target
    elif hasattr(target, "__dict__"):
        dict_attrs = target.__dict__
    else:
        return target
    result = {}
    for attr in dict_attrs:
        if isinstance(attr, str) and (not len(attr) or attr[0] == "_"):
            continue

        if isinstance(dict_attrs[attr], SavableDataset):
            sub_dict = dict_attrs[attr].state_dict()
            for sub_attr in sub_dict:
                result[f"{attr}.{sub_attr}"] = sub_dict[sub_attr]
        # We don't serialize any dataset that doesn't have this mixin.
        elif isinstance(dict_attrs[attr], Dataset):
            raise TypeError(
                "Attempting to serialize an unsupported dataset",
                attr,
                type(dict_attrs[attr]),
            )
        elif isinstance(dict_attrs[attr], dict):
            sub_dict = _state_dict_save_helper(dict_attrs[attr])
            # format like pytorch nn.Module state dicts instead of nesting
            for sub_attr in sub_dict:
                result[f"{attr}.{sub_attr}"] = sub_dict[sub_attr]
        elif isinstance(dict_attrs[attr], list):
            result[attr] = [_state_dict_save_helper(x) for x in dict_attrs[attr]]
        else:
            result[attr] = dict_attrs[attr]
    return result


def _state_dict_load_helper(target, state_dict):
    """
    A helper function to load the state of a dataset.

    Args:
        target: The object to load the state into.
        state_dict (dict): The state dictionary to load.
    """
    if isinstance(target, dict):
        dict_attrs = target
    else:
        dict_attrs = target.__dict__
    for attr in state_dict:
        if "." in attr:
            attr_name, sub_attr = attr.split(".", 1)
            sub_dict = {sub_attr: state_dict[attr]}
            if isinstance(dict_attrs[attr_name], SavableDataset):
                # Make sure we use any overriden load_state_dict in nested datasets
                dict_attrs[attr_name].load_state_dict(sub_dict)
            else:
                _state_dict_load_helper(dict_attrs[attr_name], sub_dict)
        elif attr not in dict_attrs:
            raise KeyError(f"Unexpected key {attr} in state dict")
        elif isinstance(dict_attrs[attr], SavableDataset):
            dict_attrs[attr].load_state_dict(state_dict[attr])
        elif isinstance(dict_attrs[attr], dict):
            _state_dict_load_helper(dict_attrs[attr], state_dict[attr])
        else:
            dict_attrs[attr] = state_dict[attr]


class SavableDataset:
    """
    In large-scale pre-training, because we typically only train for only a
    single epoch, we often need to be able to retain the state of the dataset
    across restarts.
    This mixin indicates a dataset that can be serialized and deserialized.
    IterableDatasets that implement this interface are expected to be
    re-startable (pick up where they left off).

    If you need a restartable MapDataSet, wrap it in a
    RestartableFromMapDataset.

    The default implementation may not work for your dataset, so override
    `state_dict` and/or `load_state_dict` as-needed.
    """

    def state_dict(self):
        """
        Get the state of the dataset.

        Returns:
            dict: The state dictionary.
        """
        return _state_dict_save_helper(self)

    def load_state_dict(self, state_dict):
        """
        Load the state of the dataset.

        Args:
            state_dict (dict): The state dictionary to load.
        """
        _state_dict_load_helper(self, state_dict)


class RestartableFromMapDataset(SavableDataset, IterableDataset):
    """
    A restartable iterable dataset from a map-style dataset.

    Args:
        map_ds (Dataset): The map-style dataset.
    """

    def __init__(self, map_ds: Dataset):
        super().__init__()
        self._map_ds = map_ds
        self.current_index = 0

    def __iter__(self):
        """
        Iterate over the dataset.
        """
        for index in range(self.current_index, len(self._map_ds)):
            self.current_index = index + 1
            yield self._map_ds[index]

    def __len__(self):
        """
        Get the number of items in the dataset.
        """
        return len(self._map_ds)


class PackedSequenceDataset(Dataset, SavableDataset):
    """
    A dataset that packs sequences to a maximum length.

    Args:
        dataset (SavableDataset): The dataset to pack.
        max_seq_len (int): The maximum sequence length.
    """

    def __init__(self, dataset: SavableDataset, max_seq_len: int):
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.buffer: List[Any] = []

    def __iter__(self):
        """
        Iterate over the dataset.
        """
        for example in self.dataset:
            self.buffer.extend(example)
            while len(self.buffer) >= self.max_seq_len:
                next_val = self.buffer[: self.max_seq_len]
                self.buffer = self.buffer[self.max_seq_len :]
                yield next_val


class WithSeparatorDataset(Dataset, SavableDataset):
    """
    A dataset that adds separator tokens to the beginning and end of each sequence.

    Args:
        dataset (SavableDataset): The dataset to wrap.
        bos_token_id (Optional[int]): The beginning-of-sequence token ID.
        eos_token_id (Optional[int]): The end-of-sequence token ID.
    """

    def __init__(
        self,
        dataset: SavableDataset,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        self.dataset = dataset
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id

    def __iter__(self):
        """
        Iterate over the dataset.
        """
        for example in self.dataset:
            result = []
            if self._bos_token_id is not None:
                result.append(self._bos_token_id)
            result.extend(example)
            if self._eos_token_id is not None:
                result.append(self._eos_token_id)
            yield result
