# DeepFilter
DeepFilter is a metaproteomics-filtering tool based on deep learning model. It is aimed at improving the  improving peptide identifications of microbial communities from a collection of tandem mass spectra. The details are available in https://arxiv.org/pdf/2009.11241.pdf

## Setup and installation
### Dependency
* python == 3.6
* numpy == 1.17.2
* scikit-learn >= 0.21.3
* pytorch(gpu version) >= 1.4.0
* CUDA Version 10.2
### Requirement
* Linux operation system
* GPU memory should be more than 8 Gb for inference mode otherwise the batchsize should be adjuested
* GPU memory should be more than 20 Gb for training mode

## Toy example of DeepFilter
### Post-processing
The toy example given is to help getting a quick start. The files of toy example include:
* testData_1.ms2 -> experimental tandem mass spectrum data
* test.data.pin -> database searching results by Comet
* temp_model/ directory -> include three models, the file "benchmark.pt" is the pre-trained model for inference
* The fasta file for filtering is attached in the link https://myunt-my.sharepoint.com/:u:/r/personal/xuan_guo_unt_edu/Documents/Shichao/Metaproteomics%20Deep%20Learning/testdata.fasta.zip?csf=1&web=1&e=c8as9q
The file inference.sh is to rescore the PSM from exsisting database searching results, the use is:
```
#!/bin/bash
./inference.sh -in testData_1.ms2 -s test.data.pin -m temp_model/benchmark.pt -o test.rescore.txt

```
*  test.rescore.txt -> The rescore results for PSMs
*  testidx.txt, testcharge.txt, testpeptide.fasta are processing files to generate isotope distribution

### Protein identification at PSM, peptide and protein level by accepting FDR equals to 1%





