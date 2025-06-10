#!/usr/bin/env python3
"""
SAP Dataset Audio Converter

This script converts multi-channel audio files in the SAP dataset to mono channel
with 16kHz sampling rate. It processes multiple subsets (Train, Dev) and creates
a parallel directory structure with '-mono' suffix.

Usage:
    python sap_mono_converter.py --input-dir /path/to/sap --subsets Train Dev
    python sap_mono_converter.py --input-dir /path/to/sap --subsets Train Dev --output-suffix mono --sample-rate 22050
"""

import os
import argparse
import subprocess
from typing import List, Tuple
from pathlib import Path

from tqdm import tqdm
from rich.console import Console
from rich.logging import RichHandler
import logging

# Configure logging with rich formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


def fast_scandir(path: str, exts: List[str], recursive: bool = True) -> Tuple[List[str], List[str]]:
    """
    Scan files recursively faster than glob.
    
    Args:
        path: Directory path to scan
        exts: List of file extensions to match (e.g., ['.wav', '.mp3'])
        recursive: Whether to scan subdirectories recursively
        
    Returns:
        Tuple of (subfolders, files) lists
        
    Note:
        Adapted from github.com/drscotthawley/aeiou/blob/main/aeiou/core.py
    """
    subfolders, files = [], []

    try:
        for f in os.scandir(path):
            try:
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in exts:
                        files.append(f.path)
            except (OSError, ValueError):
                # Handle symbolic link errors and other file access issues
                continue
    except (PermissionError, FileNotFoundError, OSError):
        logger.warning(f"Cannot access directory: {path}")
        return subfolders, files

    if recursive:
        for subfolder_path in list(subfolders):
            sf, f = fast_scandir(subfolder_path, exts, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)

    return subfolders, files


def convert_audio_to_mono(input_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """
    Convert audio file to mono channel with specified sample rate using ffmpeg.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to output audio file
        sample_rate: Target sample rate in Hz (default: 16000)
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ac', '1',  # Convert to mono (1 channel)
            '-ar', str(sample_rate),  # Set sample rate
            '-y',  # Overwrite output file if it exists
            '-loglevel', 'error',  # Reduce ffmpeg output verbosity
            output_path
        ]
        
        # Execute ffmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error for {input_path}: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error converting {input_path}: {str(e)}")
        return False


def process_subset(input_dir: Path, subset_name: str, output_suffix: str, 
                  sample_rate: int, skip_existing: bool = True) -> Tuple[int, int]:
    """
    Process a single subset directory (e.g., Train, Dev).
    
    Args:
        input_dir: Base input directory path
        subset_name: Name of the subset (e.g., 'Train', 'Dev')
        output_suffix: Suffix to add to output directory name
        sample_rate: Target sample rate for conversion
        skip_existing: Whether to skip already converted files
        
    Returns:
        Tuple of (successful_conversions, total_files)
    """
    subset_path = input_dir / subset_name
    
    if not subset_path.exists():
        logger.warning(f"Subset directory not found: {subset_path}")
        return 0, 0
    
    logger.info(f"Processing subset: {subset_name}")
    
    # Scan for audio files
    _, audio_files = fast_scandir(str(subset_path), exts=['.wav'], recursive=True)
    
    if not audio_files:
        logger.warning(f"No audio files found in {subset_path}")
        return 0, 0
    
    logger.info(f"Found {len(audio_files)} audio files in {subset_name}")
    
    successful_conversions = 0
    
    # Process each audio file
    for audio_file in tqdm(audio_files, desc=f"Converting {subset_name}"):
        # Generate output path by replacing directory name
        input_path = Path(audio_file)
        relative_path = input_path.relative_to(input_dir)
        output_path = input_dir.parent / f"{input_dir.name}-{output_suffix}" / relative_path
        
        # Skip if output file already exists and skip_existing is True
        if skip_existing and output_path.exists():
            logger.debug(f"Skipping existing file: {output_path}")
            continue
        
        # Convert audio file
        if convert_audio_to_mono(str(input_path), str(output_path), sample_rate):
            successful_conversions += 1
        else:
            logger.error(f"Failed to convert: {input_path}")
    
    return successful_conversions, len(audio_files)


def main():
    """Main function to handle command line arguments and coordinate processing."""
    parser = argparse.ArgumentParser(
        description="Convert SAP dataset audio files to mono channel",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='/home/jovyan/workspace/aura/corpus/audio/sap',
        help='Base directory containing SAP dataset subsets (default: /home/jovyan/workspace/aura/corpus/audio/sap)'
    )
    parser.add_argument(
        '--subsets',
        nargs='+',
        default=['Train', 'Dev'],
        help='List of subset names to process (default: Train Dev)'
    )
    parser.add_argument(
        '--output-suffix',
        type=str,
        default='mono',
        help='Suffix to append to output directory name (default: mono)'
    )

    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Target sample rate in Hz (default: 16000)'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Overwrite existing converted files instead of skipping them'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg is not installed or not available in PATH")
        return 1
    
    logger.info(f"Starting SAP dataset conversion")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output suffix: {args.output_suffix}")
    logger.info(f"Target sample rate: {args.sample_rate} Hz")
    logger.info(f"Subsets to process: {', '.join(args.subsets)}")
    
    total_successful = 0
    total_files = 0
    
    # Process each subset
    for subset in args.subsets:
        successful, total = process_subset(
            input_dir=input_dir,
            subset_name=subset,
            output_suffix=args.output_suffix,
            sample_rate=args.sample_rate,
            skip_existing=not args.no_skip_existing
        )
        total_successful += successful
        total_files += total
        
        if total > 0:
            success_rate = (successful / total) * 100
            logger.info(f"Subset {subset}: {successful}/{total} files converted successfully ({success_rate:.1f}%)")
    
    # Print final summary
    if total_files > 0:
        overall_success_rate = (total_successful / total_files) * 100
        console.print(f"\n[bold green]Conversion completed!")
        console.print(f"[green]Total: {total_successful}/{total_files} files converted successfully ({overall_success_rate:.1f}%)")
        
        output_dir = input_dir.parent / f"{input_dir.name}-{args.output_suffix}"
        console.print(f"[blue]Output directory: {output_dir}")
    else:
        logger.warning("No files were processed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())