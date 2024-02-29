# Latent Dynamics Networks

This repository contains codes accompanying the paper [1], introducing [**Latent Dynamics Network (LDNets)**](https://doi.org/10.1038/s41467-024-45323-x), a scientific machine learning method capable of uncovering low-dimensional intrinsic dynamics in systems exhibiting a spatio-temporal behavior in response to external stimuli. LDNets automatically discover a low-dimensional manifold while learning the system dynamics, eliminating the need for training an auto-encoder and avoiding operations in the high-dimensional space. They predict the evolution, even in time-extrapolation scenarios, of space-dependent fields without relying on predetermined grids, thus enabling weight-sharing across query-points. Lightweight and easy-to-train, LDNets demonstrate superior accuracy (normalized error 5 times smaller) in highly-nonlinear problems, with significantly fewer trainable parameters (more than 10 times fewer) compared to state-of-the-art methods.

## Requirements

To run these codes, you need Python (version 3.9) with the modules listed in the file `requirements.txt`. If you are using `pip`, you can install them by running:
```bash
pip install -r requirements.txt
```

## Downloading data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10436827.svg)](https://doi.org/10.5281/zenodo.10436827)

To replicate the results presented in the paper [1], the necessary training and testing data have been organized in a dedicated Zenodo repository. Follow the steps below to incorporate the data into your local repository:

1. Navigate to [this Zenodo repository](https://doi.org/10.5281/zenodo.10436827).

2. Download the `data.zip` file from the repository.

3. Extract the contents of `data.zip` to the root level of your local copy of this repository. This will generate a new folder named `data`.

You can also achieve this by running this script:
```bash
curl https://zenodo.org/records/10436827/files/data.zip -o data.zip
unzip data.zip -d data
rm data.zip
```

## References

[1] F. Regazzoni, S. Pagani, M. Salvador, L. Dede', A. Quarteroni, ["Learning the intrinsic dynamics of spatio-temporal processes through Latent Dynamics Networks"](https://doi.org/10.1038/s41467-024-45323-x) _Nature Communications_ (2024) 15, 1834