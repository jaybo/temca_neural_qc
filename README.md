# temca_neural_qc

Neural net to analyze and rate TEMCA QC images.


## Installation

### Create a virtual or conda environment

    conda create -n qc python=3.6 pip
    activate qc

### Install the package

    pip install git+https://github.com/jaybo/temca_neural_qc.git

## Usage

The installer creats an executable file `temca_neural_qc.exe` which can be run directly from the command line without invoking Python.  All directories are recursively searched from the root directory.  If subdirectories follow the `DONOTUSE` naming convention, then true/false positive/negative statistics are printed at the conclusion of the run.

    temca_neural_qc <root directory> <start barcode> <optional end barcode>

### example

    temca_neural_qc "C:/Users/jaybo/Google Drive/data/V1DD_409828_L1" 0 999999

## Typical output

path, confidence, 'BAD' if confidence > 0.5
(we're detecting bad montages, so values close to 1.0 are bad to the bone)

    ...
    C:\Users\jaybo\Google Drive\data\V1DD_409828_L1\TEMCA2\409828_L1_AI027_TEMCA2_02\140456, 0.10,
    C:\Users\jaybo\Google Drive\data\V1DD_409828_L1\TEMCA2\409828_L1_AI027_TEMCA2_02\140457, 0.62, BAD
    C:\Users\jaybo\Google Drive\data\V1DD_409828_L1\TEMCA2\409828_L1_AI027_TEMCA2_02\140458, 0.00,
    ...
