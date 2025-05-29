import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers.trainer_pt_utils import LabelSmoother

from ..logger import console

# Constant for the token ID to be ignored during label smoothing
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def get_all_linear_module_names(use_8bit: bool, model: torch.nn.Module) -> list:
    """
    Identify all linear module names in the model based on whether 8-bit precision is used.

    Args:
        use_8bit (bool): Flag indicating whether 8-bit precision is enabled.
        model (torch.nn.Module): The model to search for linear modules.

    Returns:
        list: A list of unique linear module names in the model.
    """
    # Determine the appropriate linear class based on precision
    linear_class = bnb.nn.Linear8bitLt if use_8bit else torch.nn.Linear

    # Set to store unique linear module names
    unique_module_names = set()

    # Iterate over all named modules in the model
    for module_name, module in model.named_modules():
        # Check if the module is an instance of the linear class
        if isinstance(module, linear_class):
            # Split the module name and add the relevant part to the set
            name_parts = module_name.split('.')
            unique_module_names.add(name_parts[0] if len(name_parts) == 1 else name_parts[-1])

    # Convert the set to a sorted list and return
    return sorted(unique_module_names)


def load_from_checkpoint(checkpoint_path: str, model: torch.nn.Module = None) -> None:
    """
    Placeholder function to load a model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module, optional): The model to load the checkpoint into. Defaults to None.

    Returns:
        None
    """
    # Functionality not implemented yet
    pass


def print_model_params(model: nn.Module) -> None:
    """
    Prints the number of trainable parameters, total parameters, and the percentage of trainable parameters
    for a given PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.
    """
    # Calculate total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_percentage = (trainable_params / total_params) * 100

    # Print the results
    console.log(
        f"trainable params: {trainable_params:,} || "
        f"all params: {total_params:,} || "
        f"trainable%: {trainable_percentage:.4f}"
    )
