# AURA

AURA (Atypical Understanding & Recognition for Accessibility) is a project dedicated to advancing speech recognition for individuals with atypical speech patterns. Leveraging the Speech Accessibility Project (SAP) dataset, we fine-tune state-of-the-art models to improve recognition accuracy and accessibility. Our approach is evaluated on both the SAP and TORGO datasets, demonstrating its robustness across diverse atypical speech scenarios.

## Setup

Setup the environment called `aura` by running the following commands:

```bash
conda update conda -y
conda create --name aura python=3.10
conda activate aura
conda install ipykernel ipywidgets -y
python -m ipykernel install --user --name aura --display-name "aura"
```

Install the required packages by running the following command:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

To run the notebook, execute the following command:

```bash
bash lora_finetune.sh
```

During training, you can check your GPU usage:

```bash
watch -n 0.5 -c gpustat -cp --color
```

Merge the LoRA weights with the base model using the following command:

```bash
bash merge_lora.sh
```

Evaluate the model on the SAP and TORGO datasets using the following command:

```bash
bash evaluate.sh
```

## Data Preparation

The data is prepared in JSONL manifest files that feed into the training and evaluation pipelines.

### Manifest File Format

Each line in the JSONL manifest file represents a single audio sample with its transcription and metadata. The format is as follows:

```json
{
  "audio": {
    "path": "/path/to/audio/file.wav"
  },
  "sentence": "Transcription of the audio file",
  "sentences": [],
  "duration": 14.07
}
```

### Preparing SAP Dataset

The SAP dataset includes recordings from individuals with atypical speech patterns. 

#### Downloading SAP Dataset

First download SAP dataset to `/path/to/sap` and extract all the tar files.

```bash
cd /path/to/sap/Train
for i in $(seq -f "%03g" 0 16); do tar xvf SpeechAccessibility_2025-03-31_$i.tar; done
tar xvf SpeechAccessibility_2025-03-31_Train_Only_Json.tar

cd /path/to/sap/Dev
for i in $(seq -f "%03g" 0 2); do tar xvf SpeechAccessibility_2025-03-31_$i.tar; done
tar xvf SpeechAccessibility_2025-03-31_Dev_Only_Json.tar
```

This will let you get a bunch of tar files, and you need to extract all of them.

```bash
for f in *.tar; do [[ $f != SpeechAccessibility_* ]] && tar xvf "$f"; done
```

#### Preprocessing SAP Audio Files

Before creating manifests, SAP audio files need to be preprocessed because some WAV files are not mono-channel. Convert them to mono with a 16kHz sample rate using:

```bash
python sap_mono_converter.py --input-dir /path/to/sap --sample-rate 16000 --output-suffix mono-16k
```

#### Creating SAP Manifests

After preprocessing, create the manifest files:

```bash
python prepare_sap.py --sap-dir /path/to/sap-mono-16k --output-dir /path/to/output
```

This will generate `train.jsonl` and `dev.jsonl` files for training and validation.

### Preparing TORGO Dataset

The TORGO dataset contains speech recordings from individuals with cerebral palsy (CP) and amyotrophic lateral sclerosis (ALS), classified by severity levels:

```bash
python prepare_torgo.py --torgo-dir /path/to/torgo --output-dir /path/to/output
```

This script processes the TORGO dataset and generates three JSONL files:
- `torgo_severe.jsonl`: Contains recordings from speakers with severe dysarthria
- `torgo_moderate.jsonl`: Contains recordings from speakers with moderate dysarthria
- `torgo_mild.jsonl`: Contains recordings from speakers with mild dysarthria

## License

This project is licensed under the MIT License.