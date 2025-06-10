#!/usr/bin/env python3
"""
Whisper Model Fine-tuning Script

This script provides functionality for fine-tuning OpenAI's Whisper models
using LoRA (Low-Rank Adaptation) or AdaLoRA techniques for efficient training.
"""

import os
import platform
import argparse
import functools
from typing import Optional, List, Union

import torch
from peft import (
    LoraConfig,
    AdaLoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import Seq2SeqTrainer as _Seq2SeqTrainer
from transformers import (
    Seq2SeqTrainingArguments,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

from aura.logger import console
from aura.data.datasets.asr_modelling import AudioDataset
from aura.callbacks.peft_callback import SavePeftModelCallback
from aura.data.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from aura.utils.data_utils import remove_punctuation, convert_to_simplified_chinese
from aura.utils.model_utils import load_from_checkpoint, print_model_params
from aura.utils.helpers import print_arguments, enable_gradient_for_output, add_argument


class Seq2SeqTrainer(_Seq2SeqTrainer):
    """
    Custom Seq2SeqTrainer with enhanced error handling.
    
    Extends the base Seq2SeqTrainer to handle batch processing errors
    gracefully by skipping problematic batches instead of crashing.
    """
    
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, tuple]:
        """
        Compute loss with error handling for problematic batches.
        
        Args:
            model: The model to compute loss for
            inputs: Input data batch
            return_outputs: Whether to return outputs along with loss
            num_items_in_batch: Number of items in the current batch
            
        Returns:
            Loss tensor or tuple of (loss, outputs) if return_outputs is True
        """
        try:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        except Exception as e:
            console.log(f"[Skipping batch due to exception] {e}")
            dummy_loss = torch.tensor(0.0, requires_grad=True).to(model.device)
            return (dummy_loss, None) if return_outputs else dummy_loss


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command line argument parser with all training parameters.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper models using LoRA/AdaLoRA techniques",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_arg = functools.partial(add_argument, argument_parser=parser)

    # Dataset configuration
    add_arg("train_data", type=str, default="dataset/train.json",
            help="Path to the training dataset JSON file")
    add_arg("test_data", type=str, default="dataset/test.json",
            help="Path to the testing dataset JSON file")
    
    # Model configuration
    add_arg("base_model", type=str, default="openai/whisper-tiny",
            help="Base Whisper model identifier from HuggingFace")
    add_arg("output_dir", type=str, default="output/",
            help="Directory to save trained models and checkpoints")
    add_arg("language", type=str, default="English",
            help="Target language for transcription/translation")
    add_arg("task", type=str, default="transcribe", choices=['transcribe', 'translate'],
            help="Task type: transcribe or translate")
    
    # Training hyperparameters
    add_arg("learning_rate", type=float, default=1e-3,
            help="Learning rate for optimizer")
    add_arg("num_train_epochs", type=int, default=3,
            help="Total number of training epochs")
    add_arg("max_steps", type=int, default=None,
            help="Maximum number of training steps (overrides epochs if set)")
    add_arg("warmup_steps", type=int, default=50,
            help="Number of warmup steps for learning rate scheduler")
    
    # Batch and gradient settings
    add_arg("per_device_train_batch_size", type=int, default=8,
            help="Training batch size per device")
    add_arg("per_device_eval_batch_size", type=int, default=8,
            help="Evaluation batch size per device")
    add_arg("gradient_accumulation_steps", type=int, default=1,
            help="Number of steps to accumulate gradients before updating")
    
    # Logging and evaluation
    add_arg("logging_steps", type=int, default=100,
            help="Number of steps between logging updates")
    add_arg("eval_steps", type=int, default=1000,
            help="Number of steps between evaluations")
    add_arg("save_steps", type=int, default=1000,
            help="Number of steps between checkpoint saves")
    add_arg("save_total_limit", type=int, default=3,
            help="Maximum number of checkpoints to keep")
    
    # Audio processing
    add_arg("min_audio_len", type=float, default=0.5,
            help="Minimum audio length in seconds")
    add_arg("max_audio_len", type=float, default=30.0,
            help="Maximum audio length in seconds")
    add_arg("timestamps", type=bool, default=False,
            help="Whether to use timestamp information during training")
    
    # LoRA configuration
    add_arg("lora_type", type=str, default='lora', choices=['lora', 'ada_lora'],
            help="Type of LoRA configuration to use")
    
    # Standard LoRA hyperparameters
    add_arg("lora_r", type=int, default=32,
            help="LoRA rank (dimension of adaptation)")
    add_arg("lora_alpha", type=int, default=64,
            help="LoRA alpha parameter for scaling")
    add_arg("lora_dropout", type=float, default=0.05,
            help="LoRA dropout rate")
    add_arg("lora_bias", type=str, default="none", choices=["none", "all", "lora_only"],
            help="LoRA bias configuration")
    
    # AdaLoRA specific hyperparameters
    add_arg("ada_lora_init_r", type=int, default=12,
            help="AdaLoRA initial rank")
    add_arg("ada_lora_target_r", type=int, default=4,
            help="AdaLoRA target rank")
    add_arg("ada_lora_beta1", type=float, default=0.85,
            help="AdaLoRA beta1 parameter")
    add_arg("ada_lora_beta2", type=float, default=0.85,
            help="AdaLoRA beta2 parameter")
    add_arg("ada_lora_tinit", type=int, default=200,
            help="AdaLoRA initial warmup steps")
    add_arg("ada_lora_tfinal", type=int, default=1000,
            help="AdaLoRA final steps for rank reduction")
    add_arg("ada_lora_deltaT", type=int, default=10,
            help="AdaLoRA steps between rank updates")
    add_arg("ada_lora_orth_reg_weight", type=float, default=0.5,
            help="AdaLoRA orthogonal regularization weight")
    
    # System and optimization settings
    add_arg("num_workers", type=int, default=8,
            help="Number of worker processes for data loading")
    add_arg("fp16", type=bool, default=True,
            help="Use mixed precision (FP16) training")
    add_arg("use_8bit", type=bool, default=False,
            help="Use 8-bit quantization for model weights")
    add_arg("use_compile", type=bool, default=False,
            help="Use PyTorch 2.0 compilation for optimization")
    add_arg("local_files_only", type=bool, default=False,
            help="Only load models from local files (no downloads)")
    
    # Data augmentation and resuming
    add_arg("augment_config_path", type=str, default=None,
            help="Path to data augmentation configuration file")
    add_arg("resume_from_checkpoint", type=str, default=None,
            help="Path to checkpoint directory for resuming training")
    
    # HuggingFace Hub integration
    add_arg("push_to_hub", type=bool, default=False,
            help="Push trained model to HuggingFace Hub")
    add_arg("hub_model_id", type=str, default=None,
            help="Model repository ID on HuggingFace Hub")
    
    # Experiment tracking
    add_arg("report_to", type=str, nargs='*', default=["wandb"],
            help="Experiment tracking platforms (wandb, tensorboard, etc.)")
    
    return parser


def configure_device_and_workers(args: argparse.Namespace) -> torch.device:
    """
    Configure compute device and adjust worker settings for platform.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configured torch device
    """
    # Adjust number of workers for Windows compatibility
    if platform.system() == "Windows":
        args.num_workers = 0
        console.log("Windows detected: Setting num_workers to 0")
    
    # Determine computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.log(f"Using device: {device}")
    
    return device


def load_datasets(args: argparse.Namespace, processor: WhisperProcessor) -> tuple:
    """
    Load and initialize training and testing datasets.
    
    Args:
        args: Parsed command line arguments
        processor: Whisper processor for audio and text processing
        
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
    
    console.log(f"Loaded {len(train_dataset)} training samples")
    console.log(f"Loaded {len(test_dataset)} testing samples")
    
    return train_dataset, test_dataset


def setup_model(args: argparse.Namespace) -> WhisperForConditionalGeneration:
    """
    Load and configure the Whisper model for training.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configured Whisper model
    """
    # Determine device mapping for distributed training
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        console.log(f"Distributed training detected: world_size={world_size}")
    
    # Load base model
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        load_in_8bit=args.use_8bit,
        device_map=device_map,
        local_files_only=args.local_files_only
    )
    
    # Configure model for generation
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # Prepare for efficient training
    model = prepare_model_for_kbit_training(model)
    
    # Register forward hook for gradient flow in multi-GPU setups
    model.model.encoder.conv1.register_forward_hook(enable_gradient_for_output)
    
    return model


def configure_lora(args: argparse.Namespace, model, train_dataset_size: int):
    """
    Configure and apply LoRA or AdaLoRA to the model.
    
    Args:
        args: Parsed command line arguments containing LoRA hyperparameters
        model: Base Whisper model
        train_dataset_size: Size of training dataset for AdaLoRA calculations
        
    Returns:
        Model with LoRA configuration applied
    """
    console.log("Configuring LoRA modules...")
    
    if args.resume_from_checkpoint:
        console.log(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        model = PeftModel.from_pretrained(
            model, args.resume_from_checkpoint, is_trainable=True
        )
    else:
        # Define target modules for LoRA adaptation
        target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
        
        if args.lora_type == 'ada_lora':
            # Calculate total training steps for AdaLoRA
            total_steps = args.num_train_epochs * train_dataset_size
            
            config = AdaLoraConfig(
                init_r=args.ada_lora_init_r,
                target_r=args.ada_lora_target_r,
                beta1=args.ada_lora_beta1,
                beta2=args.ada_lora_beta2,
                tinit=args.ada_lora_tinit,
                tfinal=args.ada_lora_tfinal,
                deltaT=args.ada_lora_deltaT,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                orth_reg_weight=args.ada_lora_orth_reg_weight,
                target_modules=target_modules,
                total_step=total_steps
            )
            console.log(f"Using AdaLoRA with {total_steps} total steps")
            console.log(f"AdaLoRA config: init_r={args.ada_lora_init_r}, target_r={args.ada_lora_target_r}")
            
        else:  # Standard LoRA
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=target_modules,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias
            )
            console.log(f"Using standard LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        
        model = get_peft_model(model, config)
    
    return model


def setup_training_arguments(args: argparse.Namespace, output_dir: str) -> Seq2SeqTrainingArguments:
    """
    Create training arguments configuration.
    
    Args:
        args: Parsed command line arguments
        output_dir: Output directory path
        
    Returns:
        Configured training arguments
    """
    # Handle report_to configuration
    report_to = args.report_to
    if isinstance(report_to, list) and len(report_to) == 1 and report_to[0].lower() == "none":
        report_to = []
    elif isinstance(report_to, str) and report_to.lower() == "none":
        report_to = []
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    training_args = Seq2SeqTrainingArguments(
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
        ddp_find_unused_parameters=(world_size > 1),
        dataloader_num_workers=args.num_workers,
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=args.push_to_hub
    )
    
    return training_args


def main():
    """
    Main function orchestrating the Whisper model fine-tuning process.
    
    This function handles the complete training pipeline:
    1. Parse command line arguments
    2. Configure system settings
    3. Load datasets and model
    4. Set up LoRA configuration
    5. Initialize trainer and run training
    6. Save final model and optionally push to hub
    """
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    print_arguments(args)
    
    # Configure system settings
    device = configure_device_and_workers(args)
    
    # Initialize Whisper processor
    console.log("Initializing Whisper processor...")
    processor = WhisperProcessor.from_pretrained(
        args.base_model,
        language=args.language,
        task=args.task,
        no_timestamps=not args.timestamps,
        local_files_only=args.local_files_only
    )
    
    # Load datasets
    console.log("Loading datasets...")
    train_dataset, test_dataset = load_datasets(args, processor)
    
    # Initialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Set up model
    console.log("Loading and configuring model...")
    model = setup_model(args)
    
    # Configure LoRA
    model = configure_lora(args, model, len(train_dataset))
    
    # Set up output directory
    if args.base_model.endswith("/"):
        args.base_model = args.base_model.rstrip("/")
    output_dir = os.path.join(args.output_dir, os.path.basename(args.base_model))
    
    # Create training arguments
    training_args = setup_training_arguments(args, output_dir)
    
    # Print model parameters on main process
    if training_args.local_rank in {0, -1}:
        print_model_params(model)
    
    # Initialize trainer
    console.log("Initializing trainer...")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
        callbacks=[SavePeftModelCallback]
    )
    
    # Configure model for training
    model.config.use_cache = False
    trainer._load_from_checkpoint = load_from_checkpoint
    
    # Start training
    console.log("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model state
    console.log("Saving final model...")
    trainer.save_state()
    model.config.use_cache = True
    
    # Save model on main process
    if training_args.local_rank in {0, -1}:
        final_checkpoint_path = os.path.join(output_dir, "checkpoint-final")
        model.save_pretrained(final_checkpoint_path)
        console.log(f"Final model saved to: {final_checkpoint_path}")
    
    # Push to HuggingFace Hub if requested
    if training_args.push_to_hub:
        hub_model_id = args.hub_model_id or output_dir
        console.log(f"Pushing model to HuggingFace Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
    
    console.log("Training completed successfully!")


if __name__ == '__main__':
    main()