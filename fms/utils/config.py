import copy
import inspect
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import TypeVar, Union


logger = logging.getLogger(__name__)

T = TypeVar("T", bound="ModelConfig")


@dataclass
class ModelConfig:
    """
    A base class for model configurations.
    This class provides methods for loading, saving, and updating model configurations.
    """

    @classmethod
    def load(cls, json_file: Union[str, os.PathLike]) -> "ModelConfig":
        """
        Load a model configuration from a JSON file.

        Args:
            json_file (Union[str, os.PathLike]): The path to the JSON file.

        Returns:
            ModelConfig: The loaded model configuration.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        json_dict = json.loads(text)

        return cls(
            **{
                k: v
                for k, v in json_dict.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def as_dict(self) -> dict:
        """
        Convert the model configuration to a dictionary.

        Returns:
            dict: The model configuration as a dictionary.
        """
        return asdict(self)

    def save(self, file_path: Union[str, os.PathLike]):
        """
        Save the model configuration to a JSON file.

        Args:
            file_path (Union[str, os.PathLike]): The path to the JSON file.
        """
        with open(file_path, "w") as f:
            json.dump(self.as_dict(), f)

    def updated(self: T, **kwargs) -> T:
        """
        Clone this ModelConfig and override the parameters of the ModelConfig specified by kwargs.

        Note: This will always return a deep copy.

        Args:
            **kwargs: The parameters to override.

        Returns:
            ModelConfig: A new instance of ModelConfig with the parameters overridden.
        """
        # create a deep copy as we don't want to modify this reference
        copied_config = copy.deepcopy(self)
        unknown_params = []
        for k, v in kwargs.items():
            if hasattr(copied_config, k):
                setattr(copied_config, k, v)
            else:
                unknown_params.append(k)
        if len(unknown_params) > 0:
            logger.info(
                f"""Found the following unknown parameters while cloning and updating the configuration: {unknown_params}"""
            )
        return copied_config
