# AURA: Atypical Understanding & Recognition for Accessibility

AURA is a project dedicated to advancing speech recognition for individuals with atypical speech patterns. Leveraging the Speech Accessibility Project (SAP) dataset, we fine-tune state-of-the-art models to improve recognition accuracy and accessibility. Our approach is evaluated on both the SAP and TORGO datasets, demonstrating its robustness across diverse atypical speech scenarios.

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