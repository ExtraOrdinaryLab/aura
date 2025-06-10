import os
import json
import argparse
import shutil
import subprocess
from typing import List, Dict
from pathlib import Path
from collections import defaultdict

import soundfile as sf
from tqdm import tqdm
from rich.console import Console

console = Console()


def check_sph2pipe_available():
    """Check if sph2pipe is installed and available"""
    try:
        # Method 1: Use which/where command
        if os.name == 'nt':  # Windows
            result = subprocess.run(['where', 'sph2pipe'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:  # Unix/Linux/MacOS
            result = subprocess.run(['which', 'sph2pipe'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            return True
        
        # Method 2: Use shutil.which
        if shutil.which('sph2pipe') is not None:
            return True
            
        return False
    except Exception:
        return False


def fast_scandir(path: str, extensions: List[str], recursive: bool = False):
    """Scan files recursively faster than glob."""
    subfolders, files = [], []

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(path):
            try:  # hope to avoid 'too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in extensions:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    if recursive:
        for path in list(subfolders):
            sf, f = fast_scandir(path, extensions, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)  # type: ignore

    return subfolders, files


def convert_sph_to_wav(audio_files: List[str], force_convert: bool = False):
    """Convert SPH files to WAV files."""
    # Check if sph2pipe is available
    if not check_sph2pipe_available():
        console.log("[bold red]Error: sph2pipe tool is not installed or cannot be found in system path[/bold red]")
        console.log("[yellow]Please install sph2pipe from: [link=https://github.com/ExtraOrdinaryLab/sph2pipe]https://github.com/ExtraOrdinaryLab/sph2pipe[/link][/yellow]")
        console.log("After installation, ensure sph2pipe is executable from command line (add to system PATH)")
        return []

    converted_files = []
    for audio_file in tqdm(audio_files, desc="Converting SPH to WAV"):
        sph_file = audio_file
        wav_file = audio_file.replace('.sph', '.wav')
        
        if os.path.exists(wav_file) and not force_convert:
            console.log(f"{wav_file} already exists. Skipping...")
        else:
            # Use subprocess to run, can capture errors
            try:
                result = subprocess.run(
                    f"sph2pipe -f wav {sph_file} > {wav_file}", 
                    shell=True, 
                    stderr=subprocess.PIPE, 
                    text=True
                )
                if result.returncode != 0:
                    console.log(f"[red]Conversion failed {sph_file}: {result.stderr}[/red]")
                    continue
                console.log(f"Converted {sph_file} -> {wav_file}")
            except Exception as e:
                console.log(f"[red]Error during conversion {sph_file}: {str(e)}[/red]")
                continue
        
        converted_files.append(wav_file)
    
    return converted_files


def collect_transcript_data(transcript_files: List[str], filename_to_filepath: Dict[str, str], speaker_to_channel: Dict[str, int]):
    """Collect and group transcript data by audio file."""
    transcript_by_audio = defaultdict(list)
    
    for transcript_file in tqdm(transcript_files, desc="Grouping transcript files"):
        audio_filename = Path(transcript_file).stem + '.wav'
        
        with open(transcript_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                split_line = line.split(" ")
                start = float(split_line[0])
                end = float(split_line[1])
                speaker = split_line[2].replace(":", "")
                channel = speaker_to_channel[speaker]
                transcript = " ".join(split_line[3:])
                
                transcript_by_audio[audio_filename].append({
                    "start": start,
                    "end": end,
                    "channel": channel,
                    "text": transcript,
                    "transcript_file": transcript_file
                })
    
    return transcript_by_audio


def process_audio_segments(transcript_by_audio: Dict[str, List[Dict]], 
                           filename_to_filepath: Dict[str, str],
                           audio_output_dir: str,
                           extract_segments: bool = True):
    """Process audio files and extract segments."""
    data_points = []
    segment_id = 0
    
    for audio_filename, segments in tqdm(transcript_by_audio.items(), desc="Processing audio files"):
        if audio_filename not in filename_to_filepath:
            console.log(f"Warning: Audio file not found {audio_filename}")
            continue
            
        audio_filepath = filename_to_filepath[audio_filename]
        
        # Only read audio file once
        try:
            audio_array, sample_rate = sf.read(audio_filepath)
        except Exception as e:
            console.log(f"Error reading {audio_filepath}: {e}")
            continue
            
        # Process all transcript segments for this audio file
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            channel = segment["channel"]
            
            # Ensure indices are within range
            start_sample = min(int(start * sample_rate), len(audio_array)-1)
            end_sample = min(int(end * sample_rate), len(audio_array))
            
            if start_sample >= end_sample:
                console.log(f"Warning: Invalid segment duration {audio_filepath}, start={start}, end={end}")
                continue
            
            # Create output filename
            segment_filename = f"segment_{segment_id:06d}.wav"
            segment_path = os.path.join(audio_output_dir, segment_filename)
            
            # Calculate duration
            duration = (end - start)
            
            # Extract and save audio segment (if needed)
            if extract_segments:
                # Extract audio segment
                segment_audio = audio_array[start_sample:end_sample]
                
                # If stereo audio, keep only the needed channel
                if len(segment_audio.shape) > 1 and segment_audio.shape[1] > 1:
                    segment_audio = segment_audio[:, channel]
                
                # Save audio segment
                sf.write(segment_path, segment_audio, sample_rate)
            
            # Add to data points
            data_point = {
                "id": f"segment_{segment_id:06d}",
                "start": start,
                "end": end,
                "channel": channel,
                "sentence": segment["text"],
                "sentences": [],
                "transcript": {"path": segment["transcript_file"]},
                "audio": {"path": audio_filepath},
                "duration": duration
            }
            
            # If segments are extracted, add segment path
            if extract_segments:
                data_point['original_audio'] = data_point['audio']
                data_point["audio"] = {"path": segment_path}
                del data_point['start'], data_point['end'], data_point['channel']
            
            data_points.append(data_point)
            segment_id += 1
    
    return data_points


def main():
    parser = argparse.ArgumentParser(description='Process Fisher corpus data')
    parser.add_argument('--audio-dir', type=str, default='/Users/yang/workspace/prepare-ldc/LDC2004S13',
                        help='Path to Fisher audio directory')
    parser.add_argument('--transcript-dir', type=str, default='/Users/yang/workspace/prepare-ldc/LDC2004T19',
                        help='Path to Fisher transcript directory')
    parser.add_argument('--output-dir', type=str, default='/Users/yang/workspace/prepare-ldc/fisher_data',
                        help='Path to output directory')
    parser.add_argument('--do-convert', action='store_true',
                        help='Whether to convert SPH files to WAV format (only need to run once)')
    parser.add_argument('--do-segment', action='store_true',
                        help='Whether to extract audio segments (IO intensive, only need to run once)')
    parser.add_argument('--force-convert', action='store_true',
                        help='Force re-conversion of existing WAV files')
    
    args = parser.parse_args()
    
    fisher_audio_dir = args.audio_dir
    fisher_transcript_dir = args.transcript_dir
    output_dir = args.output_dir
    audio_output_dir = os.path.join(output_dir, 'audio')
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if args.do_segment:
        os.makedirs(audio_output_dir, exist_ok=True)

    # Scan SPH audio files
    _, audio_files = fast_scandir(
        fisher_audio_dir, ['.sph'], 
        recursive=True
    )
    
    # Scan transcript files
    _, transcript_files = fast_scandir(
        os.path.join(fisher_transcript_dir, 'fe_03_p1_tran', 'data', 'trans'), 
        ['.txt'], 
        recursive=True
    )

    console.log(f'Found audio files: {len(audio_files)}')
    console.log(f'Found transcript files: {len(transcript_files)}')

    # Convert SPH to WAV (if needed)
    if args.do_convert:
        # Check if sph2pipe is available
        if not check_sph2pipe_available():
            console.log("[bold red]Error: sph2pipe tool is not installed or cannot be found in system path[/bold red]")
            console.log("[yellow]Please install sph2pipe from: [link=https://github.com/ExtraOrdinaryLab/sph2pipe]https://github.com/ExtraOrdinaryLab/sph2pipe[/link][/yellow]")
            console.log("After installation, ensure sph2pipe is executable from command line (add to system PATH)")
            return
        wav_files = convert_sph_to_wav(audio_files, force_convert=args.force_convert)
    else:
        # Assume WAV files already exist
        wav_files = [f.replace('.sph', '.wav') for f in audio_files]

    speaker_to_channel = {"A": 0, "B": 1}
    
    # Create mapping from filename to file path
    filename_to_filepath = {
        Path(wav_file).name: wav_file 
        for wav_file in wav_files
    }
    
    # Collect transcript data
    transcript_by_audio = collect_transcript_data(
        transcript_files, filename_to_filepath, speaker_to_channel
    )
    
    # Process audio segments
    data_points = process_audio_segments(
        transcript_by_audio, 
        filename_to_filepath,
        audio_output_dir,
        extract_segments=args.do_segment
    )

    console.log(f"Processed a total of {len(data_points)} segments")
    if data_points:
        console.log(f"Example data point: {data_points[0]}")
    
    # Save data points as JSONL file
    jsonl_path = os.path.join(output_dir, "fisher_segments.jsonl")
    with open(jsonl_path, 'w') as f:
        for data_point in data_points:
            f.write(json.dumps(data_point) + '\n')
    
    console.log(f"Data saved to {jsonl_path}")


if __name__ == '__main__':
    main()