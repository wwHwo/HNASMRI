# HNASMRI
This is the official implementation code for Hierarchical Neural Architecture Search with Adaptive Global-Local Feature Learning for Magnetic Resonance Image Reconstruction

## How to use
---
Data Preprocessing:
Data used in this work are publicly available from the IXI-T1,single channle of Calgary Campinas and The MICCAI 2013 grand challenge. 

1. run dataPreprocessing/deep.py
2. run dataPreprocessing/dataPreprocessing.py
Change the folder to a custom folder

The original project borrowed codes from https://github.com/yjump/NAS-for-CSMRI and https://github.com/quark0/darts.

This project is for research purpose and not approved for clinical use.

We conducted comparative experiments with the following methods：
1. https://github.com/yjump/NAS-for-CSMRI
2. https://github.com/puneesh00/cs-mri-gan
3. https://github.com/yangyan92/Pytorch_ADMM-CSNet
4. https://github.com/tensorlayer/DAGAN
5. https://github.com/Houruizhi/IDPCNN
6. https://github.com/estija/Co-VeGAN

For some of the experiments we used the pytorch version or reimplemented it with pytorch

Our code will be fully released after the paper is published.
