import os
import sys
import json
import mmap
import random
import struct
from typing import List, Optional, Tuple, Dict, Any

import librosa
import soundfile
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from ...logger import console


class BinaryDatasetWriter:
    """
    Writer class for creating binary dataset files.
    
    This class writes audio data and metadata to binary files with corresponding
    header files for efficient random access during training.
    """
    
    def __init__(self, file_prefix: str):
        """
        Initialize the binary dataset writer.
        
        Args:
            file_prefix: Prefix for the output files (.data and .header will be appended)
        """
        # Create corresponding data files
        self.data_file = open(file_prefix + '.data', 'wb')
        self.header_file = open(file_prefix + '.header', 'wb')
        self.data_count = 0
        self.current_offset = 0
        self.header_line = ''

    def add_data(self, data: str) -> None:
        """
        Add a data entry to the binary dataset.
        
        Args:
            data: JSON string containing audio metadata and transcription
        """
        key = str(self.data_count)
        data_bytes = bytes(data, encoding="utf8")
        
        # Write audio data to binary file
        self.data_file.write(struct.pack('I', len(key)))
        self.data_file.write(key.encode('ascii'))
        self.data_file.write(struct.pack('I', len(data_bytes)))
        self.data_file.write(data_bytes)
        
        # Write index information to header file
        self.current_offset += 4 + len(key) + 4
        self.header_line = key + '\t' + str(self.current_offset) + '\t' + str(len(data_bytes)) + '\n'
        self.header_file.write(self.header_line.encode('ascii'))
        self.current_offset += len(data_bytes)
        self.data_count += 1

    def close(self) -> None:
        """Close all open files."""
        self.data_file.close()
        self.header_file.close()


class BinaryDatasetReader:
    """
    Reader class for loading binary dataset files.
    
    This class provides efficient random access to audio data stored in binary format
    with duration filtering capabilities.
    """
    
    def __init__(self, data_header_path: str, min_duration: float = 0, max_duration: float = 30):
        """
        Initialize the binary dataset reader.
        
        Args:
            data_header_path: Path to the .header file
            min_duration: Minimum audio duration in seconds (default: 0)
            max_duration: Maximum audio duration in seconds (default: 30)
        """
        self.keys = []
        self.offset_dictionary = {}
        self.file_pointer = open(data_header_path.replace('.header', '.data'), 'rb')
        self.memory_map = mmap.mmap(self.file_pointer.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Read and filter data based on duration constraints
        for line in tqdm(open(data_header_path, 'rb'), desc='Loading data list'):
            key, value_position, value_length = line.split('\t'.encode('ascii'))
            data = self.memory_map[int(value_position):int(value_position) + int(value_length)]
            data = str(data, encoding="utf-8")
            data = json.loads(data)
            
            # Skip audio files that don't meet duration requirements
            if data["duration"] < min_duration:
                continue
            if max_duration != -1 and data["duration"] > max_duration:
                continue
                
            self.keys.append(key)
            self.offset_dictionary[key] = (int(value_position), int(value_length))

    def get_data(self, key: bytes) -> Optional[Dict[str, Any]]:
        """
        Retrieve data for a specific key.
        
        Args:
            key: The key to retrieve data for
            
        Returns:
            Dictionary containing audio metadata and transcription, or None if key not found
        """
        position_info = self.offset_dictionary.get(key, None)
        if position_info is None:
            return None
            
        value_position, value_length = position_info
        data = self.memory_map[value_position:value_position + value_length]
        data = str(data, encoding="utf-8")
        return json.loads(data)

    def get_keys(self) -> List[bytes]:
        """Get all available keys in the dataset."""
        return self.keys

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.keys)


class AudioDataset(Dataset):
    """
    Custom PyTorch Dataset for audio data processing.
    
    This dataset class handles loading, preprocessing, and augmentation of audio data
    for speech recognition model training, particularly designed for Whisper models.
    """
    
    def __init__(
        self,
        data_list_path: str,
        processor,
        is_mono: bool = True,
        language: Optional[str] = None,
        use_timestamps: bool = False,
        sample_rate: int = 16000,
        min_duration: float = 0.5,
        max_duration: float = 30,
        min_sentence_length: int = 0,
        max_sentence_length: int = 200,
        augmentation_config_path: Optional[str] = None
    ):
        """
        Initialize the audio dataset.
        
        Args:
            data_list_path: Path to data list file or binary list header file
            processor: Whisper preprocessing tool obtained from WhisperProcessor.from_pretrained
            is_mono: Whether to convert audio to mono channel (must be True)
            language: Language of the fine-tuning data
            use_timestamps: Whether to use timestamps during fine-tuning
            sample_rate: Audio sampling rate (default: 16000)
            min_duration: Minimum audio duration in seconds (cannot be less than 0.5, default: 0.5s)
            max_duration: Maximum audio duration in seconds (cannot be greater than 30, default: 30s)
            min_sentence_length: Minimum sentence character count for fine-tuning (default: 0)
            max_sentence_length: Maximum sentence character count for fine-tuning (default: 200)
            augmentation_config_path: Path to data augmentation configuration file
        """
        super(AudioDataset, self).__init__()
        
        # Validate input parameters
        assert min_duration >= 0.5, f"min_duration cannot be less than 0.5, current: {min_duration}"
        assert max_duration <= 30, f"max_duration cannot be greater than 30, current: {max_duration}"
        assert min_sentence_length >= 0, f"min_sentence_length cannot be less than 0, current: {min_sentence_length}"
        assert max_sentence_length <= 200, f"max_sentence_length cannot be greater than 200, current: {max_sentence_length}"
        
        # Initialize instance variables
        self.data_list_path = data_list_path
        self.processor = processor
        self.sample_rate = sample_rate
        self.is_mono = is_mono
        self.language = language
        self.use_timestamps = use_timestamps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        
        # Initialize vocabulary and special tokens
        self.vocabulary = self.processor.tokenizer.get_vocab()
        self.start_of_transcript_token = self.vocabulary['<|startoftranscript|>']
        self.end_of_text_token = self.vocabulary['<|endoftext|>']
        
        # Handle different model versions for special tokens
        if '<|nospeech|>' in self.vocabulary.keys():
            self.no_speech_token = self.vocabulary['<|nospeech|>']
            self.timestamp_begin_token = None
        else:
            # Compatibility with older models
            self.no_speech_token = self.vocabulary['<|nocaptions|>']
            self.timestamp_begin_token = self.vocabulary['<|notimestamps|>'] + 1
            
        self.data_list: List[dict] = []
        
        # Load data list
        self._load_data_list()
        
        # Initialize data augmentation configuration
        self.augmentation_configs = None
        self.noise_file_paths = None
        self.speed_rate_options = None
        if augmentation_config_path:
            with open(augmentation_config_path, 'r', encoding='utf-8') as file:
                self.augmentation_configs = json.load(file)

    def _load_data_list(self) -> None:
        """Load the data list from file or binary format."""
        if self.data_list_path.endswith(".header"):
            # Load binary data list
            self.dataset_reader = BinaryDatasetReader(
                data_header_path=self.data_list_path,
                min_duration=self.min_duration,
                max_duration=self.max_duration
            )
            self.data_list = self.dataset_reader.get_keys()
        else:
            # Load regular data list from JSON file
            with open(self.data_list_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            self.data_list = []
            
            for line in tqdm(lines, desc='Loading data list'):
                if isinstance(line, str):
                    line = json.loads(line)
                if not isinstance(line, dict):
                    continue
                    
                # Skip audio files that don't meet duration requirements
                if line["duration"] < self.min_duration:
                    continue
                if self.max_duration != -1 and line["duration"] > self.max_duration:
                    continue
                    
                # Skip audio files that don't meet sentence length requirements
                sentence_length = len(line["sentences"])
                if sentence_length < self.min_sentence_length or sentence_length > self.max_sentence_length:
                    continue
                    
                self.data_list.append(dict(line))

    def _get_list_data(self, index: int) -> Tuple[np.ndarray, int, Any, Optional[str]]:
        """
        Get audio data, sample rate, and text from the data list.
        
        Args:
            index: Index of the data item to retrieve
            
        Returns:
            Tuple of (audio_sample, sample_rate, transcript, language)
        """
        if self.data_list_path.endswith(".header"):
            data_entry = self.dataset_reader.get_data(self.data_list[index])
        else:
            data_entry = self.data_list[index]
            
        # Extract audio file path and transcript
        audio_file_path = data_entry["audio"]['path']
        transcript = data_entry["sentences"] if self.use_timestamps else data_entry["sentence"]
        language = data_entry.get("language", None)
        domain = data_entry.get("domain", None)
        
        # Load audio data
        if 'start_time' not in data_entry["audio"].keys():
            audio_sample, original_sample_rate = soundfile.read(audio_file_path, dtype='float32')
        else:
            start_time = data_entry["audio"]["start_time"]
            end_time = data_entry["audio"]["end_time"]
            # Load audio segment
            audio_sample, original_sample_rate = self.slice_audio_from_file(
                audio_file_path, start=start_time, end=end_time
            )
            
        audio_sample = audio_sample.T
        
        # Convert to mono channel
        if self.is_mono:
            audio_sample = librosa.to_mono(audio_sample)
            
        # Apply data augmentation
        if self.augmentation_configs:
            audio_sample, original_sample_rate = self.apply_augmentation(audio_sample, original_sample_rate)
            
        # Resample if necessary
        if self.sample_rate != original_sample_rate:
            audio_sample = self.resample_audio(
                audio_sample, 
                original_sample_rate=original_sample_rate, 
                target_sample_rate=self.sample_rate
            )
            
        return audio_sample, original_sample_rate, transcript, language, domain

    def _load_timestamps_transcript(self, transcript: List[dict]) -> Dict[str, List[int]]:
        """
        Load transcript with timestamps.
        
        Args:
            transcript: List of dictionaries containing text and timing information
            
        Returns:
            Dictionary containing processed labels with timestamps
        """
        assert isinstance(transcript, list), f"transcript should be list, current: {type(transcript)}"
        
        processed_data = dict()
        labels = self.processor.tokenizer.prefix_tokens[:3]
        
        for segment in transcript:
            # Encode target text as label IDs
            start_time = segment['start'] if round(segment['start'] * 100) % 2 == 0 else segment['start'] + 0.01
            if self.timestamp_begin_token is None:
                start_token = self.vocabulary[f'<|{start_time:.2f}|>']
            else:
                start_token = self.timestamp_begin_token + round(start_time * 100) // 2
                
            end_time = segment['end'] if round(segment['end'] * 100) % 2 == 0 else segment['end'] - 0.01
            if self.timestamp_begin_token is None:
                end_token = self.vocabulary[f'<|{end_time:.2f}|>']
            else:
                end_token = self.timestamp_begin_token + round(end_time * 100) // 2
                
            text_labels = self.processor(text=segment['text']).input_ids[4:-1]
            labels.extend([start_token])
            labels.extend(text_labels)
            labels.extend([end_token])
            
        processed_data['labels'] = labels + [self.end_of_text_token]
        return processed_data

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Args:
            index: Index of the item to retrieve
            
        Returns:
            Dictionary containing processed audio features and labels
        """
        try:
            # Get audio data, sample rate, and text from data list
            audio_sample, sample_rate, transcript, language, domain = self._get_list_data(index=index)
            
            # Set language for individual data entries
            self.processor.tokenizer.set_prefix_tokens(
                language=language if language is not None else self.language
            )
            
            if len(transcript) > 0:
                # Load transcript with timestamps
                if self.use_timestamps:
                    data = self._load_timestamps_transcript(transcript=transcript)
                    # Compute log-Mel input features from audio array
                    data["input_features"] = self.processor(
                        audio=audio_sample, 
                        sampling_rate=self.sample_rate
                    ).input_features
                else:
                    # Get log-Mel features and label IDs
                    data = self.processor(
                        audio=audio_sample, 
                        sampling_rate=self.sample_rate, 
                        text=transcript
                    )
            else:
                # Use <|nospeech|> token if no text is available
                data = self.processor(audio=audio_sample, sampling_rate=self.sample_rate)
                data['labels'] = [self.start_of_transcript_token, self.no_speech_token, self.end_of_text_token]
            
            if domain is not None:
                data['domain'] = domain

            return data
            
        except Exception as error:
            console.log(f'Error reading data at index: {index}, error message: {error}', file=sys.stderr)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data_list)

    @staticmethod
    def slice_audio_from_file(file_path: str, start: float, end: float) -> Tuple[np.ndarray, int]:
        """
        Load a segment of audio from a file.
        
        Args:
            file_path: Path to the audio file
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            Tuple of (audio_sample, sample_rate)
        """
        sound_file = soundfile.SoundFile(file_path)
        sample_rate = sound_file.samplerate
        duration = round(float(len(sound_file)) / sample_rate, 3)
        start = round(start, 3)
        end = round(end, 3)
        
        # Handle negative time values (count from end)
        if start < 0.0:
            start += duration
        if end < 0.0:
            end += duration
            
        # Ensure boundaries are valid
        if start < 0.0:
            start = 0.0
        if end > duration:
            end = duration
        if end < 0.0:
            raise ValueError(f"Slice end position ({end} s) is out of bounds")
        if start > end:
            raise ValueError(f"Slice start position ({start} s) is later than end position ({end} s)")
            
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sound_file.seek(start_frame)
        audio_sample = sound_file.read(frames=end_frame - start_frame, dtype='float32')
        
        return audio_sample, sample_rate

    def apply_augmentation(self, audio_sample: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """
        Apply data augmentation to audio sample.
        
        Args:
            audio_sample: Input audio sample
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (augmented_sample, sample_rate)
        """
        for config in self.augmentation_configs:
            if config['type'] == 'speed' and random.random() < config['prob']:
                if self.speed_rate_options is None:
                    min_speed_rate = config['params']['min_speed_rate']
                    max_speed_rate = config['params']['max_speed_rate']
                    num_rates = config['params']['num_rates']
                    self.speed_rate_options = np.linspace(min_speed_rate, max_speed_rate, num_rates, endpoint=True)
                rate = random.choice(self.speed_rate_options)
                audio_sample = self.change_audio_speed(audio_sample, speed_rate=rate)
                
            if config['type'] == 'shift' and random.random() < config['prob']:
                min_shift_ms = config['params']['min_shift_ms']
                max_shift_ms = config['params']['max_shift_ms']
                shift_ms = random.randint(min_shift_ms, max_shift_ms)
                audio_sample = self.shift_audio(audio_sample, sample_rate, shift_ms=shift_ms)
                
            if config['type'] == 'volume' and random.random() < config['prob']:
                min_gain_dbfs = config['params']['min_gain_dBFS']
                max_gain_dbfs = config['params']['max_gain_dBFS']
                gain = random.randint(min_gain_dbfs, max_gain_dbfs)
                audio_sample = self.adjust_volume(audio_sample, gain=gain)
                
            if config['type'] == 'resample' and random.random() < config['prob']:
                new_sample_rates = config['params']['new_sample_rates']
                new_sample_rate = np.random.choice(new_sample_rates)
                audio_sample = self.resample_audio(
                    audio_sample, 
                    original_sample_rate=sample_rate, 
                    target_sample_rate=new_sample_rate
                )
                sample_rate = new_sample_rate
                
            if config['type'] == 'noise' and random.random() < config['prob']:
                min_snr_db = config['params']['min_snr_dB']
                max_snr_db = config['params']['max_snr_dB']
                if self.noise_file_paths is None:
                    self.noise_file_paths = []
                    noise_directory = config['params']['noise_dir']
                    if os.path.exists(noise_directory):
                        for file in os.listdir(noise_directory):
                            self.noise_file_paths.append(os.path.join(noise_directory, file))
                noise_path = random.choice(self.noise_file_paths)
                snr_db = random.randint(min_snr_db, max_snr_db)
                audio_sample = self.add_background_noise(
                    audio_sample, sample_rate, noise_path=noise_path, snr_db=snr_db
                )
                
        return audio_sample, sample_rate

    @staticmethod
    def change_audio_speed(audio_sample: np.ndarray, speed_rate: float) -> np.ndarray:
        """
        Change the speed of audio sample.
        
        Args:
            audio_sample: Input audio sample
            speed_rate: Speed rate multiplier
            
        Returns:
            Speed-adjusted audio sample
        """
        if speed_rate == 1.0:
            return audio_sample
        if speed_rate <= 0:
            raise ValueError("Speed rate should be greater than zero")
            
        old_length = audio_sample.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        audio_sample = np.interp(new_indices, old_indices, audio_sample).astype(np.float32)
        
        return audio_sample

    @staticmethod
    def shift_audio(audio_sample: np.ndarray, sample_rate: int, shift_ms: int) -> np.ndarray:
        """
        Apply time shift to audio sample.
        
        Args:
            audio_sample: Input audio sample
            sample_rate: Sample rate of the audio
            shift_ms: Shift amount in milliseconds
            
        Returns:
            Time-shifted audio sample
        """
        duration = audio_sample.shape[0] / sample_rate
        if abs(shift_ms) / 1000.0 > duration:
            raise ValueError("Absolute value of shift_ms should be less than audio duration")
            
        shift_samples = int(shift_ms * sample_rate / 1000)
        if shift_samples > 0:
            audio_sample[:-shift_samples] = audio_sample[shift_samples:]
            audio_sample[-shift_samples:] = 0
        elif shift_samples < 0:
            audio_sample[-shift_samples:] = audio_sample[:shift_samples]
            audio_sample[:-shift_samples] = 0
            
        return audio_sample

    @staticmethod
    def adjust_volume(audio_sample: np.ndarray, gain: float) -> np.ndarray:
        """
        Adjust the volume of audio sample.
        
        Args:
            audio_sample: Input audio sample
            gain: Gain in dB
            
        Returns:
            Volume-adjusted audio sample
        """
        audio_sample *= 10.**(gain / 20.)
        return audio_sample

    @staticmethod
    def resample_audio(audio_sample: np.ndarray, original_sample_rate: int, target_sample_rate: int) -> np.ndarray:
        """
        Resample audio to a different sample rate.
        
        Args:
            audio_sample: Input audio sample
            original_sample_rate: Original sample rate
            target_sample_rate: Target sample rate
            
        Returns:
            Resampled audio sample
        """
        audio_sample = librosa.resample(
            audio_sample, 
            orig_sr=original_sample_rate, 
            target_sr=target_sample_rate
        )
        return audio_sample

    def add_background_noise(self, audio_sample: np.ndarray, sample_rate: int, 
                           noise_path: str, snr_db: float, max_gain_db: float = 300.0) -> np.ndarray:
        """
        Add background noise to audio sample.
        
        Args:
            audio_sample: Input audio sample
            sample_rate: Sample rate of the audio
            noise_path: Path to noise file
            snr_db: Signal-to-noise ratio in dB
            max_gain_db: Maximum gain in dB
            
        Returns:
            Audio sample with added noise
        """
        noise_sample, _ = librosa.load(noise_path, sr=sample_rate)
        
        # Normalize audio volume to prevent noise from being too loud
        target_db = -20
        gain = min(max_gain_db, target_db - self.calculate_rms_db(audio_sample))
        audio_sample *= 10. ** (gain / 20.)
        
        # Set noise volume
        sample_rms_db = self.calculate_rms_db(audio_sample)
        noise_rms_db = self.calculate_rms_db(noise_sample)
        noise_gain_db = min(sample_rms_db - noise_rms_db - snr_db, max_gain_db)
        noise_sample *= 10. ** (noise_gain_db / 20.)
        
        # Adjust noise length to match audio sample length
        if noise_sample.shape[0] < audio_sample.shape[0]:
            difference_duration = audio_sample.shape[0] - noise_sample.shape[0]
            noise_sample = np.pad(noise_sample, (0, difference_duration), 'wrap')
        elif noise_sample.shape[0] > audio_sample.shape[0]:
            start_frame = random.randint(0, noise_sample.shape[0] - audio_sample.shape[0])
            noise_sample = noise_sample[start_frame:audio_sample.shape[0] + start_frame]
            
        audio_sample += noise_sample
        return audio_sample

    @staticmethod
    def calculate_rms_db(audio_sample: np.ndarray) -> float:
        """
        Calculate RMS (Root Mean Square) in dB.
        
        Args:
            audio_sample: Input audio sample
            
        Returns:
            RMS value in dB
        """
        mean_square = np.mean(audio_sample ** 2)
        return 10 * np.log10(mean_square)