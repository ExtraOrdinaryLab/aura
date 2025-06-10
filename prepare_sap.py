import os
import re
import json
import argparse
from glob import glob
from pathlib import Path

import librosa
from tqdm.auto import tqdm
from rich.console import Console

console = Console()


def load_data(sap_dir, split, skip_examples: list = []):
    json_files = glob(os.path.join(sap_dir, split, '**', '*.json'), recursive=True)
    json_files = set(list(json_files))

    data_points = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        for data_point in json_data['Files']:
            filename = data_point['Filename']
            if Path(filename).name in skip_examples:
                continue
            filepath = os.path.join(sap_dir, split, filename.split('_')[0], filename)
            transcript = data_point['Prompt']['Transcript']
            data_points.append({
                'audio': filepath, 
                'text': transcript,
            })
    return data_points


def main():
    parser = argparse.ArgumentParser(
        description="Process the SAP speech database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--sap-dir", 
        type=str, 
        help="Path to the SAP directory"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Output directory for jsonl manifests"
    )
    args = parser.parse_args()

    write_train_jsonl = True
    write_dev_jsonl = True

    train_data_points = load_data(
        args.sap_dir, 
        'Train', 
    )
    dev_data_points = load_data(
        args.sap_dir, 
        'Dev', 
    )
    console.log(f"Train files: {len(train_data_points)}")
    console.log(f"Dev files: {len(dev_data_points)}")

    # Write into jsonl file
    if write_dev_jsonl:
        dev_jsonl_filepath = os.path.join(args.output_dir, 'dev.jsonl')
        with open(dev_jsonl_filepath, 'w') as f:
            for data_point in tqdm(dev_data_points):
                audio_filepath = data_point['audio']
                duration = librosa.get_duration(path=audio_filepath)
                if duration <= 0:
                    raise ValueError(f"Duration of {audio_filepath} is {duration}")
                transcript = re.sub(r'\[.*?\]', '', data_point['text']).strip()
                json_data = {
                    'audio': {'path': audio_filepath}, 
                    'sentence': transcript, 
                    'sentences': [], 
                    'duration': duration
                }
                f.write(json.dumps(json_data) + '\n')
    if write_train_jsonl:
        train_jsonl_filepath = os.path.join(args.output_dir, 'train.jsonl')
        with open(train_jsonl_filepath, 'w') as f:
            for data_point in tqdm(train_data_points):
                audio_filepath = data_point['audio']
                duration = librosa.get_duration(path=audio_filepath)
                if duration <= 0:
                    raise ValueError(f"Duration of {audio_filepath} is {duration}")
                transcript = re.sub(r'\[.*?\]', '', data_point['text']).strip()
                json_data = {
                    'audio': {'path': audio_filepath}, 
                    'sentence': transcript, 
                    'sentences': [], 
                    'duration': duration
                }
                f.write(json.dumps(json_data) + '\n')


if __name__ == '__main__':
    main()