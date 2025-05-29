"""
Whisper Model Evaluation Script

This script evaluates a fine-tuned Whisper model on a test dataset,
computing Character Error Rate (CER) and Word Error Rate (WER) metrics.
It supports both transcription and translation tasks with configurable
preprocessing options.
"""

import gc
import re
import argparse
import platform
import functools
from typing import List, Tuple

import torch
import evaluate
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from aura.logger import console
from aura.data.datasets.asr_modelling import AudioDataset
from aura.data.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from aura.utils.data_utils import remove_punctuation, convert_to_simplified_chinese
from aura.utils.helpers import add_argument, print_arguments


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up and configure the argument parser for the evaluation script.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_argument, argument_parser=parser)
    
    # Data and model paths
    add_arg("test_data_path", type=str, default="dataset/test.json",
            help="Path to the test dataset")
    add_arg("model_path", type=str, default="models/whisper-tiny-finetune",
            help="Path to the merged model or HuggingFace model name")
    
    # Evaluation parameters
    add_arg("batch_size", type=int, default=16,
            help="Batch size for evaluation")
    add_arg("num_workers", type=int, default=8,
            help="Number of threads for data loading")
    add_arg("language", type=str, default="English",
            help="Language setting (full name or abbreviation, None for multilingual)")
    
    # Text processing options
    add_arg("remove_punctuation", type=bool, default=True,
            help="Whether to remove punctuation marks")
    add_arg("convert_to_simplified", type=bool, default=True,
            help="Whether to convert to simplified Chinese")
    add_arg("use_timestamps", type=bool, default=False,
            help="Whether to use timestamp data during evaluation")
    
    # Audio filtering parameters
    add_arg("min_audio_duration", type=float, default=0.5,
            help="Minimum audio duration in seconds")
    add_arg("max_audio_duration", type=float, default=30,
            help="Maximum audio duration in seconds")
    
    # Model loading options
    add_arg("local_files_only", type=bool, default=False,
            help="Whether to load model locally only without downloading")
    add_arg("task", type=str, default="transcribe", choices=['transcribe', 'translate'],
            help="Model task type")
    add_arg("evaluation_metric", type=str, default="cer", choices=['cer', 'wer'],
            help="Evaluation metric type")
    
    return parser


def validate_model_path(model_path: str) -> None:
    """
    Validate if the model path exists or is a valid HuggingFace model name.
    
    Args:
        model_path: Path to the model or HuggingFace model name
        
    Raises:
        AssertionError: If the model path is invalid
    """
    # Note: Original validation is commented out but kept for reference
    # assert 'openai' == os.path.dirname(model_path) or os.path.exists(model_path), \
    #     f"Model file {model_path} does not exist. Please check if the model has been successfully merged " \
    #     f"or if it's a valid HuggingFace model"
    pass


def setup_processor_and_model(args: argparse.Namespace) -> Tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    """
    Initialize and configure the Whisper processor and model.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple of (processor, model)
    """
    # Get Whisper data processor (includes feature extractor and tokenizer)
    processor = WhisperProcessor.from_pretrained(
        args.model_path,
        language=args.language,
        task=args.task,
        no_timestamps=not args.use_timestamps,
        local_files_only=args.local_files_only
    )
    
    # Load the model
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        local_files_only=args.local_files_only
    )
    
    # Configure model generation settings
    model.generation_config.language = args.language.lower()
    model.generation_config.forced_decoder_ids = None
    model.eval()
    
    return processor, model


def create_test_dataloader(args: argparse.Namespace, processor: WhisperProcessor) -> DataLoader:
    """
    Create the test dataset and data loader.
    
    Args:
        args: Parsed command line arguments
        processor: Whisper processor instance
        
    Returns:
        DataLoader for the test dataset
    """
    # Create test dataset
    test_dataset = AudioDataset(
        data_list_path=args.test_data_path,
        processor=processor,
        use_timestamps=args.use_timestamps,
        min_duration=args.min_audio_duration,
        max_duration=args.max_audio_duration
    )
    console.log(f"Test dataset size: {len(test_dataset)}")
    
    # Create data collator for padding
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Create data loader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator
    )
    
    return test_dataloader


def preprocess_texts(predictions: List[str], references: List[str], args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    """
    Preprocess prediction and reference texts according to configuration.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        args: Parsed command line arguments
        
    Returns:
        Tuple of (processed_predictions, processed_references)
    """
    processed_predictions = predictions[:]
    processed_references = references[:]
    
    # Remove punctuation if specified
    if args.remove_punctuation:
        processed_predictions = remove_punctuation(processed_predictions)
        processed_references = remove_punctuation(processed_references)
    
    # Convert to simplified Chinese if specified
    if args.convert_to_simplified:
        processed_predictions = convert_to_simplified_chinese(processed_predictions)
        processed_references = convert_to_simplified_chinese(processed_references)
    
    # Convert to lowercase
    processed_predictions = [text.lower() for text in processed_predictions]
    # Remove content in brackets and convert to lowercase for references
    processed_references = [re.sub(r'\[.*?\]', '', text.lower()).strip() for text in processed_references]
    
    return processed_predictions, processed_references


def evaluate_model(model: WhisperForConditionalGeneration, 
                  test_dataloader: DataLoader, 
                  processor: WhisperProcessor, 
                  args: argparse.Namespace) -> Tuple[float, float]:
    """
    Evaluate the model on the test dataset.
    
    Args:
        model: Whisper model to evaluate
        test_dataloader: DataLoader for test data
        processor: Whisper processor
        args: Parsed command line arguments
        
    Returns:
        Tuple of (CER, WER) scores
    """
    # Initialize evaluation metrics
    cer_metric = evaluate.load('evaluate-metric/cer')
    wer_metric = evaluate.load('evaluate-metric/wer')
    
    # Start evaluation loop
    for step, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
        with torch.autocast(device_type="cuda"):
            with torch.no_grad():
                # Generate predictions
                generated_tokens = model.generate(
                    input_features=batch["input_features"].cuda(),
                    decoder_input_ids=batch["labels"][:, :4].cuda(),
                    max_new_tokens=255
                ).cpu().numpy()
                
                # Process labels
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                
                # Convert tokens to text
                decoded_predictions = processor.tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_references = processor.tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )
                
                # Preprocess texts according to configuration
                processed_predictions, processed_references = preprocess_texts(
                    decoded_predictions, decoded_references, args
                )
                
                # Add batch to metrics
                cer_metric.add_batch(predictions=processed_predictions, references=processed_references)
                wer_metric.add_batch(predictions=processed_predictions, references=processed_references)
        
        # Clean up memory
        del generated_tokens, labels, batch
        gc.collect()
    
    # Compute final metrics
    cer_score = cer_metric.compute()
    wer_score = wer_metric.compute()
    
    return cer_score, wer_score


def main() -> None:
    """
    Main function to orchestrate the evaluation process.
    """
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    print_arguments(args)
    
    # Validate model path
    validate_model_path(args.model_path)
    
    # Adjust num_workers for Windows compatibility
    if platform.system() == "Windows":
        args.num_workers = 0
        console.log("Windows detected: Setting num_workers to 0 for compatibility")
    
    # Initialize processor and model
    console.log("Loading processor and model...")
    processor, model = setup_processor_and_model(args)
    
    # Create test data loader
    console.log("Creating test data loader...")
    test_dataloader = create_test_dataloader(args, processor)
    
    # Evaluate the model
    console.log("Starting evaluation...")
    cer_score, wer_score = evaluate_model(model, test_dataloader, processor, args)
    
    # Print results
    console.log(f"\nEvaluation Results:")
    console.log(f"Word Error Rate (WER): {round(wer_score, 5)}")
    console.log(f"Character Error Rate (CER): {round(cer_score, 5)}")


if __name__ == '__main__':
    main()