# DeepFilter
DeepFilter is a metaproteomics-filtering tool based on deep learning model. It is aimed at improving the  improving peptide identifications of microbial communities from a collection of tandem mass spectra. The details are available in https://arxiv.org/pdf/2009.11241.pdf

## Setup and installation
#### Dependency
* python >= 3.6
* numpy >= 1.17.2
* pytorch(gpu version) >= 1.4.0
* CUDA Version 10.2
#### Requirement
* Linux operation system
* GPU memory should be more than 8 Gb for inference mode otherwise the batchsize should be adjuested
* GPU memory should be more than 20 Gb for training mode

## Toy example of DeepFilter
The toy example given is to help getting a quick start. The files of toy example include:
* testData_1.ms2 -> experimental tandem mass spectrum data
* test.data.pin -> database searching results by Comet
* temp_model/ directory -> include three models, the file "benchmark.pt" is the pre-trained model for inference
* 



