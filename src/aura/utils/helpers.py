import os
import hashlib
import tarfile
import urllib.request

import torch
import torch.nn as nn
from tqdm import tqdm

from ..logger import console


def print_arguments(arguments):
    """
    Print the configuration arguments in a formatted manner.
    """
    console.log("-----------  Configuration Arguments -----------")
    for argument, value in vars(arguments).items():
        console.log(f"{argument}: {value}")
    console.log("------------------------------------------------")


def string_to_boolean(value: str):
    """
    Convert a string representation of a boolean value to a boolean.
    """
    value = value.lower()
    if value in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif value in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"Invalid truth value: {value!r}")


def string_to_none(value):
    """
    Convert the string 'None' to a NoneType value, otherwise return the original string.
    """
    return None if value == 'None' else value


def add_argument(argument_name, type, default, help, argument_parser, **kwargs):
    """
    Add an argument to the argument parser with the specified properties.
    """
    type = string_to_boolean if type == bool else type
    type = string_to_none if type == str else type
    argument_parser.add_argument(
        f"--{argument_name}",
        default=default,
        type=type,
        help=f"{help} Default: %(default)s.",
        **kwargs
    )


def calculate_md5(file_path):
    """
    Calculate the MD5 checksum of a file.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url, expected_md5sum, target_directory):
    """
    Download a file from the specified URL to the target directory and verify its MD5 checksum.

    Args:
        url (str): The URL of the file to download.
        expected_md5sum (str): The expected MD5 checksum of the file.
        target_directory (str): The directory to save the downloaded file.

    Returns:
        str: The path to the downloaded file.
    """
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    file_name = os.path.basename(url)
    file_path = os.path.join(target_directory, file_name)

    if not (os.path.exists(file_path) and calculate_md5(file_path) == expected_md5sum):
        console.log(f"Downloading {url} to {file_path} ...")
        with urllib.request.urlopen(url) as source, open(file_path, "wb") as output:
            total_size = int(source.info().get("Content-Length"))
            with tqdm(total=total_size, ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as progress_bar:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break
                    output.write(buffer)
                    progress_bar.update(len(buffer))
        console.log(f"\nVerifying MD5 checksum of {file_path} ...")
        if calculate_md5(file_path) != expected_md5sum:
            raise RuntimeError("MD5 checksum verification failed.")
    else:
        console.log(f"File already exists, skipping download: {file_path}")
    return file_path


def unpack_archive(file_path, target_directory, remove_archive=False):
    """
    Unpack a tar archive to the specified directory.

    Args:
        file_path (str): The path to the tar archive.
        target_directory (str): The directory to extract the archive to.
        remove_archive (bool): Whether to remove the archive file after extraction.
    """
    console.log(f"Unpacking {file_path} ...")
    with tarfile.open(file_path) as tar:
        tar.extractall(target_directory)
    if remove_archive:
        os.remove(file_path)


def enable_gradient_for_output(module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    """
    Enable the requires_grad attribute for the output tensor.

    Args:
        module: The module that produced the output tensor.
        input_tensor: The input tensor to the module.
        output_tensor: The output tensor from the module.
    """
    output_tensor.requires_grad_(True)