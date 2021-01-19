# FCNGRU
FCNA

Locating Transcription factor binding sites (TFBSs) by fully convolutional network
Requirements

    Pytorch 1.1
    Python 3.6
    CUDA 9.0
    Python packages: biopython, sklearn

Data preparation

(1) Downloading 45 in invitro from The UniPROBE database is available at http://uniprobe.org , and put it into /your path/invitro/.

(2) Pre-processing datasets.

    Usage:
    
    data_normal.py
    
    then

    bash process.sh <data path>

Implementation

Running FCNGRU

    Usage:

    bash run.sh <data path>


Predicting motifs

    Usage:

    bash motif.sh <data path> <trained model path>


