#!/usr/bin/env python3
"""
Whisper Model Fine-tuning Script

This script provides functionality for fine-tuning OpenAI's Whisper model
on custom datasets for automatic speech recognition (ASR) tasks.
"""

import argparse
import functools
import os
import platform
from typing import Optional, Union

import torch
from transformers import (
    Seq2SeqTrainer as _Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor
)

from aura.logger import console
from aura.data.datasets.asr_modelling import AudioDataset
from aura.callbacks.peft_callback import SavePeftModelCallback
from aura.data.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from aura.utils.data_utils import remove_punctuation, convert_to_simplified_chinese
from aura.utils.model_utils import load_from_checkpoint, print_model_params
from aura.utils.helpers import print_arguments, enable_gradient_for_output, add_argument


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up and configure the argument parser with all training parameters."""
    parser = argparse.ArgumentParser(
        description="Script for fine-tuning a Whisper model on custom datasets."
    )
    add_arg = functools.partial(add_argument, argument_parser=parser)
    
    # Data configuration
    add_arg("train_data", type=str, default="dataset/train.json",
            help="Path to the training dataset")
    add_arg("test_data", type=str, default="dataset/test.json",
            help="Path to the test dataset")
    add_arg("augment_config_path", type=str, default=None,
            help="Path to data augmentation configuration file")
    
    # Model configuration
    add_arg("base_model", type=str, default="openai/whisper-tiny",
            help="Base Whisper model to fine-tune")
    add_arg("language", type=str, default="Chinese",
            help="Language setting (full name or abbreviation). "
                 "Set to None for multilingual training")
    add_arg("task", type=str, default="transcribe",
            choices=['transcribe', 'translate'], help="Model task type")
    add_arg("local_files_only", type=bool, default=False,
            help="Whether to load model from local files only")
    
    # Training parameters
    add_arg("output_dir", type=str, default="output/",
            help="Directory to save trained models")
    add_arg("num_train_epochs", type=int, default=3,
            help="Number of training epochs")
    add_arg("max_steps", type=int, default=None,
            help="Maximum number of training steps")
    add_arg("learning_rate", type=float, default=1e-5,
            help="Learning rate (smaller for full parameter fine-tuning)")
    add_arg("per_device_train_batch_size", type=int, default=8,
            help="Training batch size per device")
    add_arg("per_device_eval_batch_size", type=int, default=8,
            help="Evaluation batch size per device")
    add_arg("gradient_accumulation_steps", type=int, default=1,
            help="Number of gradient accumulation steps")
    add_arg("warmup_steps", type=int, default=50,
            help="Number of warmup steps")
    
    # Logging and evaluation
    add_arg("logging_steps", type=int, default=100,
            help="Number of steps between logging")
    add_arg("eval_steps", type=int, default=1000,
            help="Number of steps between evaluations")
    add_arg("save_steps", type=int, default=1000,
            help="Number of steps between model saves")
    add_arg("save_total_limit", type=int, default=10,
            help="Maximum number of checkpoints to keep")
    add_arg("report_to", type=str, nargs='*', default=["wandb"],
            help="List of integrations to report results and logs to. "
                 "Supported platforms: 'wandb', 'tensorboard', 'comet_ml', 'mlflow', 'neptune', 'clearml'. "
                 "Use 'none' to disable reporting. Example: --report_to wandb tensorboard")
    
    # Audio processing
    add_arg("min_audio_len", type=float, default=0.5,
            help="Minimum audio length in seconds")
    add_arg("max_audio_len", type=float, default=30,
            help="Maximum audio length in seconds (cannot exceed 30)")
    add_arg("timestamps", type=bool, default=False,
            help="Whether to use timestamp data during training")
    
    # System configuration
    add_arg("num_workers", type=int, default=8,
            help="Number of data loading workers")
    add_arg("fp16", type=bool, default=True,
            help="Whether to use FP16 training")
    add_arg("use_8bit", type=bool, default=False,
            help="Whether to quantize model to 8-bit")
    add_arg("use_compile", type=bool, default=False,
            help="Whether to use PyTorch 2.0 compiler")
    
    # Checkpoint and Hub
    add_arg("resume_from_checkpoint", type=str, default=None,
            help="Path to checkpoint for resuming training")
    add_arg("push_to_hub", type=bool, default=False,
            help="Whether to push model weights to HuggingFace Hub")
    add_arg("hub_model_id", type=str, default=None,
            help="HuggingFace Hub model repository ID")
    
    return parser


class Seq2SeqTrainer(_Seq2SeqTrainer):
    """Custom Seq2Seq trainer with enhanced error handling."""
    
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute loss with exception handling to skip problematic batches.
        
        Args:
            model: The model to compute loss for
            inputs: Input data dictionary
            return_outputs: Whether to return outputs along with loss
            num_items_in_batch: Number of items in the batch
            
        Returns:
            Loss tensor or tuple of (loss, outputs) if return_outputs=True
        """
        try:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        except Exception as e:
            console.log(f"[Skipping batch due to exception] {e}")
            dummy_loss = torch.tensor(0.0, requires_grad=True).to(model.device)
            return (dummy_loss, None) if return_outputs else dummy_loss


def load_datasets(args: argparse.Namespace, processor: WhisperProcessor) -> tuple:
    """
    Load and prepare training and test datasets.
    
    Args:
        args: Command line arguments
        processor: Whisper processor for data processing
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_dataset = AudioDataset(
        data_list_path=args.train_data,
        processor=processor,
        language=args.language,
        use_timestamps=args.timestamps,
        min_duration=args.min_audio_len,
        max_duration=args.max_audio_len,
        augmentation_config_path=args.augment_config_path
    )
    
    test_dataset = AudioDataset(
        data_list_path=args.test_data,
        processor=processor,
        language=args.language,
        use_timestamps=args.timestamps,
        min_duration=args.min_audio_len,
        max_duration=args.max_audio_len
    )
    
    console.log(f"Training data: {len(train_dataset)}, Test data: {len(test_dataset)}")
    return train_dataset, test_dataset


def setup_model(args: argparse.Namespace) -> WhisperForConditionalGeneration:
    """
    Set up and configure the Whisper model for training.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configured Whisper model
    """
    # Configure device mapping for distributed training
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    
    # Load base model
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        load_in_8bit=args.use_8bit,
        device_map=device_map,
        local_files_only=args.local_files_only
    )
    
    # Configure model for training
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # Register forward hook for multi-GPU training
    model.model.encoder.conv1.register_forward_hook(enable_gradient_for_output)
    
    # Load from checkpoint if specified
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        console.log(f"Loading model from checkpoint: {args.resume_from_checkpoint}")
        model = WhisperForConditionalGeneration.from_pretrained(
            args.resume_from_checkpoint,
            load_in_8bit=args.use_8bit,
            device_map=device_map,
            local_files_only=args.local_files_only
        )
    
    return model


def create_training_arguments(args: argparse.Namespace) -> Seq2SeqTrainingArguments:
    """
    Create training arguments configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        Training arguments object
    """
    # Clean up base model path
    if args.base_model.endswith("/"):
        args.base_model = args.base_model[:-1]
    
    output_dir = os.path.join(args.output_dir, os.path.basename(args.base_model))
    
    # Configure distributed training
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    
    # Handle report_to configuration
    report_to = args.report_to
    if isinstance(report_to, list) and len(report_to) == 1 and report_to[0].lower() == "none":
        report_to = []
    elif isinstance(report_to, str) and report_to.lower() == "none":
        report_to = []
    
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_strategy="steps",
        eval_strategy="steps",
        load_best_model_at_end=True,
        fp16=args.fp16,
        report_to=report_to,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        torch_compile=args.use_compile,
        save_total_limit=args.save_total_limit,
        optim='adamw_torch',
        ddp_find_unused_parameters=False if ddp else None,
        dataloader_num_workers=args.num_workers,
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=args.push_to_hub,
    )


def print_model_info(model: WhisperForConditionalGeneration, local_rank: int) -> None:
    """
    Print model parameter information.
    
    Args:
        model: The model to analyze
        local_rank: Local rank for distributed training
    """
    if local_rank == 0 or local_rank == -1:
        console.log('=' * 90)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        console.log(f"Trainable parameters: {trainable_params:,} "
              f"({trainable_params/all_params:.2%} of {all_params:,})")
        console.log('=' * 90)


def main() -> None:
    """Main training function."""
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    print_arguments(args)
    
    # Adjust num_workers for Windows
    if platform.system() == "Windows":
        args.num_workers = 0
    
    # Initialize processor
    processor = WhisperProcessor.from_pretrained(
        args.base_model,
        language=args.language,
        task=args.task,
        no_timestamps=not args.timestamps,
        local_files_only=args.local_files_only
    )
    
    # Load datasets
    train_dataset, test_dataset = load_datasets(args, processor)
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Setup model
    model = setup_model(args)
    console.log('Using full parameter fine-tuning...')
    
    # Create training arguments
    training_args = create_training_arguments(args)
    
    # Print model information
    print_model_info(model, training_args.local_rank)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor
    )
    
    # Configure model for training
    model.config.use_cache = False
    trainer._load_from_checkpoint = load_from_checkpoint
    
    # Start training
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    trainer.save_state()
    model.config.use_cache = True
    
    # Save final checkpoint
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        final_output_dir = os.path.join(training_args.output_dir, "checkpoint-final")
        model.save_pretrained(final_output_dir)
    
    # Push to hub if requested
    if training_args.push_to_hub:
        hub_model_id = (args.hub_model_id 
                       if args.hub_model_id is not None 
                       else training_args.output_dir)
        model.push_to_hub(hub_model_id)


if __name__ == '__main__':
    main()