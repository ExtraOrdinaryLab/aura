import os
import sys
import json
import hashlib
import tarfile
import argparse
import subprocess
from pathlib import Path

import librosa
import pandas as pd
from tqdm import tqdm
from rich.console import Console

console = Console()


class TORGOProcessor:
    """TORGO database processor for speech data preparation."""
    
    # Speaker definitions
    DYSARTHRIC_SPEAKERS = ['F01', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04', 'M05']
    CONTROL_SPEAKERS = ['FC01', 'FC02', 'FC03', 'MC01', 'MC02', 'MC03', 'MC04']
    
    # Audio processing parameters
    DURATION_MIN = 0.4
    DURATION_MAX = 60.0
    
    # Filter words for transcript cleaning
    FILTER_WORDS = [
        'JPG', 'AHPEEE', 'PAHTAHKAH', 'EEEPAH', 'FOR 5 SECONDS',
        'HE WILL ALLOW A RARE', "SAY 'OA' AS IN COAT", 'EEE', 'SAY OA',
        'RELAX YOUR MOUTH', 'XXX'
    ]
    
    # Punctuation to remove from transcripts
    PUNCTUATION = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    
    def __init__(self, torgo_dir, output_dir):
        self.torgo_dir = Path(torgo_dir)
        self.output_dir = Path(output_dir)
        self.torgo_out = self.output_dir
        
        # Create output directories
        self.torgo_out.mkdir(parents=True, exist_ok=True)
        
        self.removed_files = []
    
    def download_torgo(self):
        """Download and extract TORGO database if not present."""
        if self.torgo_dir.exists():
            console.log(f"Directory 'TORGO' already exists in {self.torgo_dir}. Skipping download.")
            return
        
        console.log(f"Directory 'TORGO' does not exist. Creating and downloading...")
        self.torgo_dir.mkdir(parents=True, exist_ok=True)
        
        urls = [
            "https://www.cs.toronto.edu/~complingweb/data/TORGO/F.tar.bz2",
            "https://www.cs.toronto.edu/~complingweb/data/TORGO/FC.tar.bz2",
            "https://www.cs.toronto.edu/~complingweb/data/TORGO/M.tar.bz2",
            "https://www.cs.toronto.edu/~complingweb/data/TORGO/MC.tar.bz2"
        ]
        
        for url in urls:
            filename = Path(url).name
            file_path = self.torgo_dir / filename
            
            console.log(f"Downloading {url}...")
            subprocess.run(["wget", "-q", "-O", str(file_path), url], check=True)
            
            console.log(f"Extracting {filename}...")
            if tarfile.is_tarfile(file_path):
                with tarfile.open(file_path, 'r:bz2') as tar:
                    tar.extractall(path=self.torgo_dir)
            else:
                console.log(f"Error: {filename} is not a valid tar.bz2 file.")
            
            console.log(f"Removing {filename}...")
            file_path.unlink()
        
        console.log("Download and extraction completed.")
    
    def get_file_lists(self):
        """Get lists of WAV and TXT files."""
        wav_files = list(self.torgo_dir.glob("**/*.wav"))
        txt_files = list(self.torgo_dir.glob("**/*.txt"))
        
        console.log(f"Found {len(wav_files)} .wav files")
        console.log(f"Found {len(txt_files)} .txt files")
        
        return wav_files, txt_files
    
    def extract_labels(self, txt_files):
        """Extract labels from transcript files."""
        text_labels = {}
        
        for txt_file in sorted(txt_files):
            # Parse file path to get tag
            parts = str(txt_file).split('.')
            temp = " ".join(parts[:-1])
            path_parts = temp.split('/')
            tag = "".join(path_parts[:-2])  # rootdirSpkSessionN
            tag2 = tag + path_parts[-1]     # rootdirSpkSessionNWAV_ID
            
            # Read and process transcript
            with open(txt_file, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip("\n")
                line = line.translate(str.maketrans('', '', self.PUNCTUATION))
                line = line.upper()
                text_labels[tag2] = line
        
        return text_labels
    
    @staticmethod
    def generate_file_id(file_path):
        """Generate a 16-character alphanumeric hash ID from file path."""
        # Use the full file path as input for consistent hashing
        hash_input = str(file_path).encode('utf-8')
        hash_object = hashlib.md5(hash_input)
        # Get first 16 characters of hex digest (numbers + a-f)
        return hash_object.hexdigest()[:16]
    
    def create_manifest(self, wav_files, text_labels):
        """Create CSV manifest from audio files and labels."""
        # Load substitution dictionary if it exists
        subs_dict = {}
        subs_file = Path('subs.json')
        if subs_file.exists():
            with open(subs_file, 'r') as f:
                subs_dict = json.load(f)
        else:
            subs_dict = {
                "READ AS IN I CAN READ": "READ", 
                "LEAD AS IN I WILL LEAD YOU": "LEAD", 
                "TEAR AS IN TEAR UP THAT PAPER": "TEAR", 
                "TEAR AS IN TEAR IN MY EYE": "TEAR"
            }
        
        csv_data = []
        
        for audio_file in sorted(wav_files):
            # Parse file path
            parts = str(audio_file).split('.')
            temp = " ".join(parts[:-1])
            path_parts = temp.split('/')
            tag = "".join(path_parts[:-2])
            tag2 = tag + path_parts[-1]
            spk = path_parts[-4]
            mic = path_parts[-2].split('_')[-1]
            
            speaker = f"TORGO_{spk}"
            corpus = 'TORGO' if spk in self.DYSARTHRIC_SPEAKERS else 'TORGO_control'
            
            if tag2 in text_labels:
                label = text_labels[tag2]
                if label in subs_dict:
                    label = subs_dict[label]
                
                csv_data.append({
                    'wav': str(audio_file),
                    'speaker': speaker,
                    'corpus': corpus,
                    'label': label,
                    'ID': self.generate_file_id(audio_file),
                    'mic': mic,
                    'length': 0.0  # Will be filled during audio check
                })
            else:
                self.removed_files.append(f"{audio_file}|None|No label")
        
        return pd.DataFrame(csv_data)
    
    def filter_transcripts(self, df):
        """Filter out unwanted transcripts."""
        console.log('Pre-processing transcripts...')
        
        filtered_df = df[~df['label'].str.contains('|'.join(self.FILTER_WORDS), case=False, na=False)]
        
        # Log filtered files
        for _, row in df.iterrows():
            if any(word in row['label'] for word in self.FILTER_WORDS):
                self.removed_files.append(f"{row['wav']}|{row['label']}|filter word")
        
        return filtered_df
    
    def check_audio_files(self, df):
        """Check audio files for corruption and duration constraints."""
        console.log(f'Checking {len(df)} audio files...')
        console.log(f'Removing files with duration <{self.DURATION_MIN}s or >{self.DURATION_MAX}s')
        
        valid_indices = []
        
        with tqdm(total=len(df), desc="Checking audio files") as pbar:
            for idx, row in df.iterrows():
                audio_path = Path(row['wav'])
                
                # Check if file exists
                if not audio_path.exists():
                    self.removed_files.append(f"{audio_path}|{row['label']}|file not found")
                    pbar.update(1)
                    continue
                
                # Check if file is empty
                if audio_path.stat().st_size == 0:
                    self.removed_files.append(f"{audio_path}|{row['label']}|empty file")
                    pbar.update(1)
                    continue
                
                try:
                    # Load audio and check duration
                    y, sr = librosa.load(str(audio_path), sr=None)
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    if duration < self.DURATION_MIN:
                        self.removed_files.append(f"{audio_path}|{row['label']}|<{self.DURATION_MIN}s duration")
                    elif duration > self.DURATION_MAX:
                        self.removed_files.append(f"{audio_path}|{row['label']}|>{self.DURATION_MAX}s duration")
                    else:
                        df.at[idx, 'length'] = duration
                        valid_indices.append(idx)
                
                except Exception as e:
                    self.removed_files.append(f"{audio_path}|{row['label']}|{str(e)}")
                
                pbar.update(1)
        
        return df.loc[valid_indices].reset_index(drop=True)
    
    def generate_report(self, df):
        """Generate summary report."""
        console.log('Creating summary report...')
        
        report = []
        report.append(f"Total number of wav files: {len(df)}")
        report.append(f"Unique labels: {len(df['label'].unique())}\n")
        
        for corpus in df['corpus'].unique():
            corpus_data = df[df['corpus'] == corpus]
            report.append(f"Number of wav files in {corpus}: {len(corpus_data)}")
        
        report.append('\n')
        
        for speaker in df['speaker'].unique():
            speaker_data = df[df['speaker'] == speaker]
            report.append(f"Number of wav files for {speaker}: {len(speaker_data)}, "
                         f"unique labels: {len(speaker_data['label'].unique())}")
        
        # Count removal reasons
        reason_counts = {}
        for item in self.removed_files:
            reason = item.strip().split('|')[-1]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        report.append(f"\nTotal number of wav files removed: {len(self.removed_files)}")
        for reason, count in reason_counts.items():
            report.append(f"{reason}: {count}")
        
        # Save report
        with open(self.torgo_out / 'TORGO_report.txt', 'w') as f:
            f.write('\n'.join(report))
    
    def save_outputs(self, df):
        """Save processed data and summaries."""
        # Save main CSV
        df.to_csv(self.torgo_out / 'TORGO.csv', index=False)
        
        # Save unique labels
        unique_labels = sorted(df['label'].unique())
        with open(self.torgo_out / 'TORGO_unique_labels.txt', 'w') as f:
            f.write('\n'.join(unique_labels))
        
        # Save removed files
        with open(self.torgo_out / 'TORGO_removed.txt', 'w') as f:
            f.write('\n'.join(self.removed_files))
        
        console.log(f"Processed {len(df)} files successfully")
        console.log(f"Results saved to {self.torgo_out}")
    
    def process(self):
        """Main processing pipeline."""
        console.log("Starting TORGO processing...")
        
        # Step 1: Download data if needed
        self.download_torgo()
        
        # Step 2: Get file lists
        wav_files, txt_files = self.get_file_lists()
        
        # Step 3: Extract labels
        text_labels = self.extract_labels(txt_files)
        
        # Step 4: Create manifest
        df = self.create_manifest(wav_files, text_labels)
        
        # Step 5: Filter transcripts
        df = self.filter_transcripts(df)
        
        # Step 6: Check audio files
        df = self.check_audio_files(df)
        
        # Step 7: Generate report
        self.generate_report(df)
        
        # Step 8: Save outputs
        self.save_outputs(df)
        
        return df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Process the TORGO speech database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--torgo-dir", 
        type=str, 
        help="Path to the TORGO directory (will be created and populated if it doesn't exist)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Output directory for CSV manifests and summaries"
    )
    
    args = parser.parse_args()
    
    # Create processor and run
    processor = TORGOProcessor(args.torgo_dir, args.output_dir)
    df = processor.process()
    
    console.log("Processing completed successfully!")

    torgo_df = pd.read_csv(os.path.join(args.output_dir, 'TORGO.csv'))
    console.log(torgo_df)

    data_points = []
    for idx, row in df.iterrows():
        audio_filepath = row['wav']
        speaker = str(row['speaker']).split('_')[-1]
        group = 'atypical' if row['corpus'] == 'TORGO' else 'control'
        text = row['label']
        mic = row['mic']
        duration = row['length']
        json_data = {
            'audio': {'path': audio_filepath}, 
            'sentence': text, 
            'sentences': [], 
            'duration': duration, 
            'mic': mic, 
            'speaker': speaker, 
            'group': group, 
        }
        data_points.append(json_data)

    with open(os.path.join(args.output_dir, 'torgo_severe.jsonl'), 'w') as f:
        for data_point in data_points:
            dysarthric_speakers = ['F01', 'M01', 'M02', 'M04']
            if data_point['speaker'] in dysarthric_speakers:
                json_data = {
                    'audio': data_point['audio'], 
                    'sentence': data_point['sentence'], 
                    'sentences': [], 
                    'duration': data_point['duration']
                }
                f.write(json.dumps(json_data) + '\n')

    with open(os.path.join(args.output_dir, 'torgo_moderate.jsonl'), 'w') as f:
        for data_point in data_points:
            dysarthric_speakers = ['M05', 'F03']
            if data_point['speaker'] in dysarthric_speakers:
                json_data = {
                    'audio': data_point['audio'], 
                    'sentence': data_point['sentence'], 
                    'sentences': [], 
                    'duration': data_point['duration']
                }
                f.write(json.dumps(json_data) + '\n')

    with open(os.path.join(args.output_dir, 'torgo_mild.jsonl'), 'w') as f:
        for data_point in data_points:
            dysarthric_speakers = ['F04', 'M03']
            if data_point['speaker'] in dysarthric_speakers:
                json_data = {
                    'audio': data_point['audio'], 
                    'sentence': data_point['sentence'], 
                    'sentences': [], 
                    'duration': data_point['duration']
                }
                f.write(json.dumps(json_data) + '\n')


if __name__ == "__main__":
    main()