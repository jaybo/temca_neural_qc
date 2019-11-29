# temca_neural_qc

Neural net to analyze TEMCA QC images


## Installation

### Create a virtual or conda environment

    conda create -n qc
    activate qc

### Locally clone the package

    git clone https://github.com/jaybo/temca_neural_qc.git mydir

    
### Install the package in developer mode

    cd mydir
    pip install -e .

## Usage

    python temca_neural_qc.py <root directory> <start barcode> <optional end barcode>

### example

    python temca_neural_qc.py "C:/Users/jaybo/Google Drive/data/V1DD_409828_L1" 0 999999

## Typical output

path, confidence, 'BAD' <if confidence > 0.5>

    ...
    C:\Users\jaybo\Google Drive\data\V1DD_409828_L1\TEMCA2\409828_L1_AI027_TEMCA2_02\140456 0.10
    C:\Users\jaybo\Google Drive\data\V1DD_409828_L1\TEMCA2\409828_L1_AI027_TEMCA2_02\140457 0.62 BAD
    C:\Users\jaybo\Google Drive\data\V1DD_409828_L1\TEMCA2\409828_L1_AI027_TEMCA2_02\140458 0.00
    ...
