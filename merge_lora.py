import os
import argparse
import functools

from peft import PeftModel, PeftConfig
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperProcessor,
)

from src.aura.utils.helpers import print_arguments, add_argument

def main():
    """
    Main function to merge a fine-tuned LoRA model with the base Whisper model,
    save the merged model, and related components (tokenizer, processor, etc.)
    to the specified directory.
    """

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Merge LoRA model with base Whisper model and save it.")
    add_arg = functools.partial(add_argument, argument_parser=parser)

    # Define command-line arguments
    add_arg("lora_model", type=str, default="output/whisper-tiny/checkpoint-best/",
            help="Path to the fine-tuned LoRA model.")
    add_arg("output_dir", type=str, default="models/",
            help="Directory to save the merged model.")
    add_arg("local_files_only", type=bool, default=False,
            help="Whether to load models only from local files without attempting to download.")

    # Parse arguments
    args = parser.parse_args()
    print_arguments(args)

    # Check if the LoRA model path exists
    assert os.path.exists(args.lora_model), f"LoRA model path '{args.lora_model}' does not exist."

    # Load LoRA configuration
    lora_config = PeftConfig.from_pretrained(args.lora_model)

    # Load the base Whisper model
    base_model = WhisperForConditionalGeneration.from_pretrained(
        lora_config.base_model_name_or_path,
        device_map={"": "cpu"},
        local_files_only=args.local_files_only,
    )

    # Merge the LoRA model with the base model
    merged_model = PeftModel.from_pretrained(
        base_model,
        args.lora_model,
        local_files_only=args.local_files_only,
    )

    # Load feature extractor, tokenizer, and processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        lora_config.base_model_name_or_path,
        local_files_only=args.local_files_only,
    )
    tokenizer = WhisperTokenizerFast.from_pretrained(
        lora_config.base_model_name_or_path,
        local_files_only=args.local_files_only,
    )
    processor = WhisperProcessor.from_pretrained(
        lora_config.base_model_name_or_path,
        local_files_only=args.local_files_only,
    )

    # Merge model parameters and set the model to evaluation mode
    merged_model = merged_model.merge_and_unload()
    merged_model.train(False)

    # Prepare the save directory
    base_model_name = lora_config.base_model_name_or_path.rstrip("/")
    save_directory = os.path.join(args.output_dir, f"{os.path.basename(base_model_name)}-finetune")
    os.makedirs(save_directory, exist_ok=True)

    # Save the merged model and related components
    merged_model.save_pretrained(save_directory, max_shard_size="4GB")
    feature_extractor.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    processor.save_pretrained(save_directory)

    print(f"Merged model saved to: {save_directory}")


if __name__ == "__main__":
    main()
