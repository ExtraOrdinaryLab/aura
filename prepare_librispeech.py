import os
import json
import argparse
from glob import glob
from pathlib import Path

import librosa
from tqdm.auto import tqdm
from rich.console import Console

console = Console()


def load_data(librispeech_dir, split):
    txt_files = glob(os.path.join(librispeech_dir, split, '**', '*.trans.txt'), recursive=True)
    txt_files = set(list(txt_files))

    data_points = []
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    id_, transcript = line.split(" ", 1)
                    audio_filename = f"{id_}.flac"
                    speaker_id, chapter_id = [int(el) for el in id_.split("-")[:2]]
                    audio_filepath = os.path.join(
                        librispeech_dir, split, str(speaker_id), str(chapter_id), audio_filename
                    )
                    data_points.append({
                        'audio': audio_filepath, 
                        'text': transcript,
                    })
    return data_points


def main():
    parser = argparse.ArgumentParser(
        description="Process the LibriSpeech speech database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--librispeech-dir", 
        type=str, 
        help="Path to the LibriSpeech directory"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Output directory for jsonl manifests"
    )
    args = parser.parse_args()

    test_clean_data_points = load_data(
        args.librispeech_dir, 
        'test-clean', 
    )
    test_other_data_points = load_data(
        args.librispeech_dir, 
        'test-other', 
    )
    console.log(f"test-clean files: {len(test_clean_data_points)}")
    console.log(f"test-other files: {len(test_other_data_points)}")

    # Write into jsonl file
    with open(os.path.join(args.output_dir, 'test_clean.jsonl'), 'w') as f:
        for data_point in tqdm(test_clean_data_points):
            audio_filepath = data_point['audio']
            duration = librosa.get_duration(path=audio_filepath)
            json_data = {
                'audio': {'path': audio_filepath}, 
                'sentence': data_point['text'], 
                'sentences': [], 
                'duration': duration
            }
            f.write(json.dumps(json_data) + '\n')
    with open(os.path.join(args.output_dir, 'test_other.jsonl'), 'w') as f:
        for data_point in tqdm(test_other_data_points):
            audio_filepath = data_point['audio']
            duration = librosa.get_duration(path=audio_filepath)
            json_data = {
                'audio': {'path': audio_filepath}, 
                'sentence': data_point['text'], 
                'sentences': [], 
                'duration': duration
            }
            f.write(json.dumps(json_data) + '\n')


if __name__ == '__main__':
    main()