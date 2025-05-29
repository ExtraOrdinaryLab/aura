import os
import platform
import argparse
import functools

from rich.console import Console

import torch
from peft import (
    LoraConfig, 
    AdaLoraConfig, 
    PeftModel, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from transformers import (
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    WhisperProcessor, 
    WhisperForConditionalGeneration
)

from src.aura.data.datasets.asr_modelling import AudioDataset
from src.aura.callbacks.peft_callback import SavePeftModelCallback
from src.aura.data.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.aura.utils.data_utils import remove_punctuation, convert_to_simplified_chinese
from src.aura.utils.model_utils import load_from_checkpoint, print_model_params
from src.aura.utils.helpers import print_arguments, enable_gradient_for_output, add_argument

# Initialize console for logging
console = Console()

# Initialize argument parser
parser = argparse.ArgumentParser(description="Script for fine-tuning a Whisper model.")
add_arg = functools.partial(add_argument, argument_parser=parser)

# Add command-line arguments
add_arg("train_data", type=str, default="dataset/train.json", help="Path to the training dataset.")
add_arg("test_data", type=str, default="dataset/test.json", help="Path to the testing dataset.")
add_arg("base_model", type=str, default="openai/whisper-tiny", help="Base Whisper model to fine-tune.")
add_arg("output_dir", type=str, default="output/", help="Directory to save trained models.")
add_arg("warmup_steps", type=int, default=50, help="Number of warmup steps during training.")
add_arg("logging_steps", type=int, default=100, help="Frequency of logging during training.")
add_arg("eval_steps", type=int, default=1000, help="Frequency of evaluation during training.")
add_arg("save_steps", type=int, default=1000, help="Frequency of saving checkpoints.")
add_arg("num_workers", type=int, default=8, help="Number of worker threads for data loading.")
add_arg("learning_rate", type=float, default=1e-3, help="Learning rate for training.")
add_arg("min_audio_len", type=float, default=0.5, help="Minimum audio length in seconds.")
add_arg("max_audio_len", type=float, default=30, help="Maximum audio length in seconds.")
add_arg("lora_type", type=str, default='lora', help="Type of LoRA configuration.")
add_arg("fp16", type=bool, default=True, help="Use FP16 precision for training.")
add_arg("use_8bit", type=bool, default=False, help="Quantize model to 8-bit.")
add_arg("timestamps", type=bool, default=False, help="Use timestamp data during training.")
add_arg("use_compile", type=bool, default=False, help="Use PyTorch 2.0 compiler.")
add_arg("local_files_only", type=bool, default=False, help="Load models only from local files.")
add_arg("num_train_epochs", type=int, default=3, help="Number of training epochs.")
add_arg("max_steps", type=int, default=None, help="Maximum number of training steps.")
add_arg("language", type=str, default="English", help="Language setting for the model.")
add_arg("task", type=str, default="transcribe", choices=['transcribe', 'translate'], help="Task for the model.")
add_arg("augment_config_path", type=str, default=None, help="Path to data augmentation config file.")
add_arg("resume_from_checkpoint", type=str, default=None, help="Path to checkpoint for resuming training.")
add_arg("per_device_train_batch_size", type=int, default=8, help="Batch size for training.")
add_arg("per_device_eval_batch_size", type=int, default=8, help="Batch size for evaluation.")
add_arg("gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
add_arg("push_to_hub", type=bool, default=False, help="Push model weights to HuggingFace Hub.")
add_arg("hub_model_id", type=str, default=None, help="Model repository ID on HuggingFace Hub.")
add_arg("save_total_limit", type=int, default=3, help="Limit the number of saved checkpoints.")

args = parser.parse_args()
print_arguments(args)

# Adjust num_workers for Windows
if platform.system() == "Windows":
    args.num_workers = 0

# Determine the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
console.log(f"Device: {device}")

def main():
    """Main function for fine-tuning the Whisper model."""
    # Initialize Whisper processor (feature extractor and tokenizer)
    processor = WhisperProcessor.from_pretrained(
        args.base_model,
        language=args.language,
        task=args.task,
        no_timestamps=not args.timestamps,
        local_files_only=args.local_files_only
    )

    # Load training and testing datasets
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
    console.log(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    # Initialize data collator for padding
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Load Whisper model
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}

    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model,
        load_in_8bit=args.use_8bit,
        device_map=device_map,
        local_files_only=args.local_files_only
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Register forward hook for multi-GPU training
    model.model.encoder.conv1.register_forward_hook(enable_gradient_for_output)

    # Load or configure LoRA modules
    console.log("Loading LoRA modules...")
    if args.resume_from_checkpoint:
        console.log("Resuming training from checkpoint.")
        model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
    else:
        target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
        if args.lora_type == 'ada_lora':
            total_steps = args.num_train_epochs * len(train_dataset)
            config = AdaLoraConfig(
                init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200,
                tfinal=1000, deltaT=10, lora_alpha=32, lora_dropout=0.1,
                orth_reg_weight=0.5, target_modules=target_modules, total_step=total_steps
            )
        elif args.lora_type == 'lora':
            config = LoraConfig(
                r=32, lora_alpha=64, target_modules=target_modules,
                lora_dropout=0.05, bias="none"
            )
        model = get_peft_model(model, config)

    # Configure output directory
    if args.base_model.endswith("/"):
        args.base_model = args.base_model.rstrip("/")
    output_dir = os.path.join(args.output_dir, os.path.basename(args.base_model))

    # Define training arguments
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
        report_to=["tensorboard"],
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

    if training_args.local_rank in {0, -1}:
        # console.log("=" * 90)
        # model.print_trainable_parameters()
        # console.log("=" * 90)
        print_model_params(model)

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
        callbacks=[SavePeftModelCallback]
    )
    model.config.use_cache = False
    trainer._load_from_checkpoint = load_from_checkpoint

    # Start training
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save the final model
    trainer.save_state()
    model.config.use_cache = True
    if training_args.local_rank in {0, -1}:
        model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))

    # Push model to HuggingFace Hub if required
    if training_args.push_to_hub:
        hub_model_id = args.hub_model_id or output_dir
        model.push_to_hub(hub_model_id)


if __name__ == '__main__':
    main()
